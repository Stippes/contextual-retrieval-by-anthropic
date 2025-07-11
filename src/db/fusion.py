from __future__ import annotations
from typing import Sequence, List
from llama_index.core.schema import NodeWithScore


def reciprocal_rank_fusion(
    vector_nodes: Sequence[NodeWithScore],
    bm25_nodes: Sequence[NodeWithScore],
    weight_vector: float = 0.8,
    weight_bm25: float = 0.2,
    k: int = 60,
) -> List[NodeWithScore]:
    """Fuse results from vector and BM25 retrieval using Reciprocal Rank Fusion."""
    # keep first occurrence of a node so we preserve provided objects
    id_to_node = {}
    for n in list(vector_nodes) + list(bm25_nodes):
        id_to_node.setdefault(n.node.node_id, n)

    # initialize scores
    scores = {node_id: 0.0 for node_id in id_to_node}

    for rank, node in enumerate(vector_nodes, start=1):
        scores[node.node.node_id] += weight_vector / (k + rank)
    for rank, node in enumerate(bm25_nodes, start=1):
        scores[node.node.node_id] += weight_bm25 / (k + rank)

    fused = [NodeWithScore(node=n.node, score=scores[nid])
             for nid, n in id_to_node.items()]
    fused.sort(key=lambda n: n.get_score(), reverse=True)
    return fused
