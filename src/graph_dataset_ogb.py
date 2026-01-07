import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, Dataset, Batch
from typing import Callable, List, Optional
from tqdm.auto import tqdm
import os
import shutil
import copy
import random

# from graph_aug import mask_edges, mask_nodes

def get_atom_feature_dims():
    """OGB 원자 특징의 각 차원에 대한 허용 가능한 값의 개수를 반환합니다."""
    return [119, 4, 12, 12, 10, 6, 6, 2, 2] # 총 9개 특징

def get_bond_feature_dims():
    """OGB 결합 특징의 각 차원에 대한 허용 가능한 값의 개수를 반환합니다."""
    return [5, 6, 2] # 총 3개 특징

def atom_to_feature_vector(atom: Chem.rdchem.Atom) -> List[int]:
    """
    RDKit의 Atom 객체를 OGB 표준 9차원 정수 벡터로 변환합니다.
    """
    # 9가지 특징 초기화
    feature = [
        # 1. Atomic Symbol (원자 종류)
        atom.GetAtomicNum(),
        # 2. Chirality (카이랄성)
        int(atom.GetChiralTag()),
        # 3. Degree (결합 차수)
        atom.GetTotalDegree(),
        # 4. Formal Charge (형식 전하)
        atom.GetFormalCharge(),
        # 5. Num Radical Electrons (라디칼 전자 수)
        atom.GetNumRadicalElectrons(),
        # 6. Hybridization (혼성 오비탈)
        int(atom.GetHybridization()),
        # 7. Is Aromatic (방향족 여부)
        atom.GetIsAromatic(),
        # 8. Is in Ring (고리 내부 원자 여부)
        atom.IsInRing(),
        # 9. Total Num Hs (총 수소 원자 수)
        atom.GetTotalNumHs(),
    ]
    return feature

def bond_to_feature_vector(bond: Chem.rdchem.Bond) -> List[int]:
    """
    RDKit의 Bond 객체를 OGB 표준 3차원 정수 벡터로 변환합니다.
    """
    # 3가지 특징 초기화
    feature = [
        # 1. Bond Type (결합 종류)
        int(bond.GetBondTypeAsDouble()),
        # 2. Is Conjugated (컨쥬게이션 결합 여부)
        bond.GetIsConjugated(),
        # 3. Is in Ring (고리 내부 결합 여부)
        bond.IsInRing(),
    ]
    # BondType을 OGB의 매핑(1,2,3,4)에 맞게 조정
    bond_map = {1.0: 1, 1.5: 4, 2.0: 2, 3.0: 3}
    feature[0] = bond_map.get(feature[0], 0) # 맵에 없으면 0

    # Stereo 정보 추가
    stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOZ: 2,
        Chem.rdchem.BondStereo.STEREOE: 3,
        Chem.rdchem.BondStereo.STEREOCIS: 4,
        Chem.rdchem.BondStereo.STEREOTRANS: 5,
    }
    feature.append(stereo_map.get(bond.GetStereo(), 0))

    final_feature = [
        feature[0], # BondType
        feature[2], # Stereo
        feature[1], # IsInRing
    ]
    return final_feature


def smiles_to_ogb_graph(smiles: str, label: Optional[List[float]] = None) -> Data:
    """
    하나의 SMILES 문자열을 OGB와 완벽히 호환되는 PyG Data 객체로 변환합니다.

    Args:
        smiles (str): 변환할 분자의 SMILES 문자열.
        label (Optional[List[float]]): 그래프의 타겟 레이블.

    Returns:
        torch_geometric.data.Data: 변환된 그래프 데이터 객체.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # 유효하지 않은 SMILES는 건너뜀

    # 1. 노드(원자) 특징 추출
    atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.long)

    # 2. 엣지(결합) 인덱스 및 특징 추출
    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # OGB 호환 엣지 특징 추출
        edge_feature = bond_to_feature_vector(bond)

        # 무방향 그래프를 위해 양방향으로 추가
        edge_indices.extend([[i, j], [j, i]])
        edge_features_list.extend([edge_feature, edge_feature])

    # 엣지가 없는 분자(예: 단일 원자) 처리
    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_feature_dims())), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 3. 그래프 레이블(y) 추가
    if label is not None:
        # OGB는 y를 [1, num_tasks] shape으로 저장합니다.
        data.y = torch.tensor([label], dtype=torch.float32)

    return data


# 각 원자 특징 차원의 어휘 크기 (0부터 시작하므로 크기 자체가 ID가 됨)
ATOM_FEATURE_VOCAB_SIZES = get_atom_feature_dims()

# [MASK] 노드의 특징 벡터
# 각 특징 차원의 어휘 크기를 마스크 ID로 사용하여, 기존 값과 겹치지 않게 합니다.
# 모델의 임베딩 레이어는 이 ID들을 처리할 수 있도록 크기가 1 더 커야 합니다.
MASK_NODE_FEATURE_VECTOR = torch.tensor(ATOM_FEATURE_VOCAB_SIZES, dtype=torch.long)

# --------------------------------------------------------------------------
# 3. 그래프 증강(Augmentation) 함수 구현
# --------------------------------------------------------------------------
def mask_nodes(graph: Batch, mask_ratio: float) -> Batch:
    if mask_ratio == 0.0:
        return graph

    num_nodes = graph.num_nodes
    num_to_mask = int(num_nodes * mask_ratio)

    if num_to_mask == 0:
        return graph  # 마스킹할 노드가 없는 경우

    # 마스킹할 노드의 인덱스를 무작위로 선택
    device = graph.x.device
    perm = torch.randperm(num_nodes, device=device)
    masked_node_indices = perm[:num_to_mask]

    # 선택된 노드의 특징 벡터를 [MASK] 토큰으로 교체
    # MASK_NODE_FEATURE_VECTOR를 그래프와 동일한 장치로 이동
    mask_feature = MASK_NODE_FEATURE_VECTOR.to(device)
    graph.x[masked_node_indices] = mask_feature

    return graph


def mask_edges(graph: Batch, mask_ratio: float) -> Batch:
    if mask_ratio == 0.0:
        return graph

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    num_edges = graph.num_edges

    if num_edges == 0:
        return graph

    # 중요: edge_index가 (i,j)와 (j,i) 쌍으로 구성되어 있다고 가정합니다.
    # 총 2E개의 엣지가 있다면, E개의 고유한 엣지 쌍이 존재합니다.
    num_edge_pairs = num_edges // 2
    num_pairs_to_remove = int(num_edge_pairs * mask_ratio)

    if num_pairs_to_remove == 0:
        return graph

    device = edge_index.device

    # 제거할 엣지 쌍의 인덱스를 무작위로 선택
    perm = torch.randperm(num_edge_pairs, device=device)
    pairs_to_remove_idx = perm[:num_pairs_to_remove]

    # 선택된 엣지 쌍과 그에 대응하는 반대 방향 엣지를 모두 제거하기 위한 마스크 생성
    # 가정: 0부터 E-1까지가 정방향, E부터 2E-1까지가 역방향 인덱스
    # 이 가정은 대부분의 PyG 데이터 처리에서 유효합니다.
    reverse_pairs_to_remove_idx = pairs_to_remove_idx + num_edge_pairs
    indices_to_remove = torch.cat([pairs_to_remove_idx, reverse_pairs_to_remove_idx])

    # 제거할 엣지를 제외하고 남길 엣지만 선택하는 마스크 생성
    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
    keep_mask[indices_to_remove] = False

    # 마스크를 사용하여 edge_index와 edge_attr 업데이트
    graph.edge_index = edge_index[:, keep_mask]
    if edge_attr is not None:
        graph.edge_attr = edge_attr[keep_mask]

    return graph



# --------------------------------------------------------------------------
# 2. InMemoryDataset 클래스 구현
# OGB 데이터셋처럼 사전 처리 후 파일로 저장하여 빠르게 로딩합니다.
# --------------------------------------------------------------------------

class OGBCompatibleSmilesDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        smiles_list: List[str],
        labels_list: Optional[List[List[float]]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        SMILES 리스트로부터 OGB와 완벽히 호환되는 PyG InMemoryDataset을 생성합니다.

        Args:
            root (str): 데이터셋이 저장될 루트 디렉토리.
            smiles_list (List[str]): 그래프로 변환할 SMILES 문자열의 리스트.
            labels_list (Optional[List[List[float]]]): 각 SMILES에 대한 레이블 리스트.
            force_reload (bool): True이면, 기존에 처리된 데이터를 삭제하고 새로 생성.
        """
        self.smiles_list = smiles_list
        self.labels_list = labels_list

        # force_reload=True 이면 기존 'processed' 폴더 삭제
        processed_dir = os.path.join(root, 'processed')
        if force_reload and os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # raw 데이터가 파일이 아닌 메모리에 있으므로 비워둡니다.
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # OGB와 유사하게 처리된 데이터를 저장할 파일 이름.
        return ['geometric_data_processed.pt']

    def download(self):
        # 다운로드할 것이 없으므로 pass
        pass

    def process(self):
        print("SMILES를 OGB 호환 그래프로 변환 중...")
        data_list = []

        iterator = self.smiles_list
        if self.labels_list is not None:
            iterator = zip(self.smiles_list, self.labels_list)
        else:
            # 레이블이 없으면 None으로 채워진 리스트를 만듦
            iterator = zip(self.smiles_list, [None] * len(self.smiles_list))

        for smiles, label in tqdm(iterator, total=len(self.smiles_list)):
            data = smiles_to_ogb_graph(smiles, label)
            if data is not None:
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"총 {len(data_list)}개의 유효한 그래프를 처리했습니다.")
        data, slices = self.collate(data_list)

        print(f"처리된 데이터를 '{self.processed_paths[0]}'에 저장 중...")
        torch.save((data, slices), self.processed_paths[0])
        print("저장 완료.")

class OnTheFlyOGBCompatibleSmilesDataset(Dataset):
    def __init__(
        self,
        root: str, # Dataset 클래스는 root를 필요로 하지만 여기선 사용하지 않음
        smiles_list: List[str],
        labels_list: Optional[List[List[float]]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_augment=False,
    ):
        """
        SMILES 리스트로부터 OGB와 호환되는 그래프를 실시간으로 생성하는 Dataset.

        Args:
            root (str): 데이터셋 루트 디렉토리 (형식상 필요).
            smiles_list (List[str]): 그래프로 변환할 SMILES 문자열의 리스트.
            labels_list (Optional[List[List[float]]]): 각 SMILES에 대한 레이블 리스트.
        """
        self.smiles_list = smiles_list
        self.labels_list = labels_list
        self.is_augment = is_augment
        super().__init__(root, transform, pre_transform)

    def len(self) -> int:
        """데이터셋의 전체 크기를 반환합니다."""
        return len(self.smiles_list)

    def get(self, idx: int) -> Data:
        """
        주어진 인덱스(idx)에 해당하는 SMILES를 그래프로 변환하여 반환합니다.
        이 메서드는 DataLoader에 의해 호출됩니다.
        """
        smiles = self.smiles_list[idx]
        label = self.labels_list[idx] if self.labels_list else None

        # 핵심 로직: 그래프를 실시간으로 생성
        original_data = smiles_to_ogb_graph(smiles, label)

        # 유효하지 않은 SMILES 처리
        if original_data is None:
            # None을 반환하면 DataLoader가 이를 건너뛰도록 처리할 수 있지만,
            # 일반적으로는 유효한 데이터만 리스트에 남기는 것이 좋습니다.
            # 여기서는 빈 Data 객체를 반환하여 오류를 방지합니다.
            print(f"경고: index {idx}의 SMILES '{smiles}'가 유효하지 않아 빈 그래프를 반환합니다.")
            return Data()

        if self.is_augment:
            mask_graph1 = mask_edges(mask_nodes(copy.deepcopy(original_data), 0.3), 0.15)
            mask_graph2 = mask_edges(mask_nodes(copy.deepcopy(original_data), 0.15), 0.3)
            return mask_graph1, mask_graph2
        else:
            return original_data

    # --- InMemoryDataset과 달리 아래 메서드들은 비워둡니다 ---
    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return []

    def download(self):
        pass

    def process(self):
        pass