import os
os.environ["RDKIT_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]   = "1"

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import PNA, global_mean_pool
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_ema import ExponentialMovingAverage
from rdkit import Chem

from src import OnTheFlyOGBCompatibleSmilesDataset
# from graph_aug import mask_edges, mask_nodes

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        if smile != canon_smile:
            print(f'{smile} > {canon_smile}')
        return canon_smile
    except:
        return np.nan

class pna4pretrain(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, deg: torch.Tensor, edge_dim: int):
        super(pna4pretrain, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.encoder = PNA(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=None,
            dropout=0.1,
            act='relu',
            norm=None, # 'BatchNorm'
            jk='cat',
            # PNA 필수 인자들
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
        )
        self.pool = global_mean_pool

        # GNN으로 얻은 그래프 벡터의 표현력을 높이기 위한 MLP
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # GNN의 hidden_dim과 동일한 입력 차원
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch):
        node_features = batch.x.float()
        edge_features = batch.edge_attr.float() if batch.edge_attr is not None else None

        node_repr = self.encoder(
            x=node_features,
            edge_index=batch.edge_index,
            edge_attr=edge_features,
            batch=batch.batch
        )
        graph_repr = self.pool(node_repr, batch.batch)

        return self.proj(graph_repr)


def info_nce_loss(z1: torch.Tensor,
                  z2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """
    SimCLR식 NT-Xent.
    z1, z2 : [B, D] (ℓ2-정규화된 임베딩)
    """
    B, _ = z1.size()
    z = torch.cat([z1, z2], dim=0)          # [2B, D]
    z = F.normalize(z, dim=1)

    # 코사인 유사도 행렬
    sim = torch.mm(z, z.t()) / temperature   # [2B, 2B]

    # (i,i) 자기 자신 제거
    diag_mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(diag_mask, -float('inf'))

    # 각 행 i의 positive index: (i+B) mod 2B
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


def main():
    CFG = {
        'EPOCHS': 30,
        'LEARNING_RATE': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'BATCH_SIZE': 2048,
        'SEED': 2025,
        'USE_AMP': True,
    }
    seed_everything(CFG['SEED'])

    # --- Data Loading ---
    data = pd.read_parquet('../data/pretrain/ZINC_canonical_cleaned.parquet')
    dataset = OnTheFlyOGBCompatibleSmilesDataset(
        root='zinc_dataset_validation',
        smiles_list=data['canonical_smiles'].to_list(),
        labels_list=None,
        is_augment=True,
    )
    loader = DataLoader(
        dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        drop_last=True,
    )

    deg = torch.tensor([2, 91814868, 285339541, 160229300, 6893150, 0, 0, 0, 0, 0])

    # --- 4. 모델 학습에 사용 ---
    # 계산된 deg 텐서는 이제 PNA 모델 초기화에 사용할 수 있습니다.
    INPUT_DIM = dataset.num_node_features  # 9
    EDGE_DIM = dataset.num_edge_features  # 3
    HIDDEN_DIM = 128
    NUM_LAYERS = 5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = pna4pretrain(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        deg=deg.to(DEVICE),
        edge_dim=EDGE_DIM
    ).to(DEVICE)

    print(f"Using device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(model)

    # --- Training Loop ---
    print(f'\nTraining Start\n\n')

    save_dir = './model_weights'
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    if CFG['USE_AMP']:
        scaler = torch.amp.GradScaler()

    best_epoch = 0
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        running_loss, n_steps = 0.0, 0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(loader, dynamic_ncols=True, leave=False)
        # for batch in pbar:
        for mask_graph1, mask_graph2 in pbar:
            optimizer.zero_grad()
            # batch = batch.to(DEVICE)

            mask_graph1 = mask_graph1.to(DEVICE)
            mask_graph2 = mask_graph2.to(DEVICE)

            # mask_graph1 = mask_edges(mask_nodes(copy.deepcopy(batch), 0.3), 0.15)
            # mask_graph2 = mask_edges(mask_nodes(copy.deepcopy(batch), 0.15), 0.3)

            if CFG['USE_AMP']:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z1 = model(mask_graph1)
                    z2 = model(mask_graph2)
                    loss = info_nce_loss(z1, z2)

                scaler.scale(loss).backward()  # backward
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = model(mask_graph1)
                z2 = model(mask_graph2)

                loss = info_nce_loss(z1, z2)

                loss.backward()
                optimizer.step()

            ema.update()
            running_loss += loss.item()
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                lr=f"{current_lr:.2e}",
                refresh=False,
            )

        train_loss = running_loss / n_steps
        # scheduler.step()

        print(f"[Epoch {epoch:02d}] loss = {running_loss / n_steps:.4f}")

        # 10epoch마다 주기적 체크포인트
        if epoch % 10 == 0:
            ckpt_name = f"pna_{HIDDEN_DIM}_{NUM_LAYERS}_epoch{epoch:02d}.pt"
            save_path_periodic = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), save_path_periodic)

        #  성능 최고 모델 저장
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            save_path = os.path.join(save_dir, f'pna_{HIDDEN_DIM}_{NUM_LAYERS}.pt')
            best_model_state = model.state_dict()
            # torch.save(model.state_dict(), save_path) # 나중에 load할 때, encoder쪽만 load
            print(f"New best model saved to {save_path} with loss {best_loss:.4f}")

    if best_model_state is not None:
        save_path = os.path.join(save_dir, f'pna_{HIDDEN_DIM}_{NUM_LAYERS}_epoch{best_epoch}.pt')
        torch.save(best_model_state, save_path)  # 나중에 load할 때, encoder쪽만 load
        print(f"Best model saved to {save_path} with Epoch {best_epoch} loss {best_loss:.4f}")



if __name__ == '__main__':
    print('\n현재 작업 디렉토리:', os.getcwd())
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f'사용 가능한 cpu 코어: {os.cpu_count()}\n')

    main()