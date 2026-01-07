import torch
import copy
from torch_geometric.data import Data, Batch

from graph_dataset_ogb import get_atom_feature_dims, get_bond_feature_dims

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
    """
    그래프의 노드 중 일부를 무작위로 선택하여 특징을 마스킹합니다.

    Args:
        graph (Batch): 증강을 적용할 PyG 배치 그래프 객체.
        mask_ratio (float): 전체 노드 중 마스킹할 비율 (0.0 ~ 1.0).

    Returns:
        Batch: 노드 특징이 마스킹된 새로운 배치 그래프 객체.
    """
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
    """
    그래프의 엣지 중 일부를 무작위로 제거합니다.
    무방향 그래프의 대칭성을 유지하면서 엣지를 제거합니다.

    Args:
        graph (Batch): 증강을 적용할 PyG 배치 그래프 객체.
        mask_ratio (float): 전체 엣지 쌍 중 제거할 비율 (0.0 ~ 1.0).

    Returns:
        Batch: 엣지가 제거된 새로운 배치 그래프 객체.
    """
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
