import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.db.fusion import reciprocal_rank_fusion
from src.db.simple_schema import TextNode, NodeWithScore


def _make_node(node_id: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=f"doc {node_id}", id_=node_id))


def test_rrf_order():
    vector_nodes = [_make_node("A"), _make_node("B"), _make_node("C")]
    bm25_nodes = [_make_node("C"), _make_node("D"), _make_node("A")]
    fused = reciprocal_rank_fusion(vector_nodes, bm25_nodes)
    ids = [n.node.node_id for n in fused]
    assert ids == ["A", "C", "B", "D"]


def test_rrf_improves_accuracy():
    vector_nodes = [_make_node("B"), _make_node("C"), _make_node("A")]
    bm25_nodes = [_make_node("C"), _make_node("D"), _make_node("A")]
    fused = reciprocal_rank_fusion(vector_nodes, bm25_nodes)

    def rank(doc_id, nodes):
        return [n.node.node_id for n in nodes].index(doc_id) + 1

    assert rank("C", fused) < rank("C", vector_nodes)
