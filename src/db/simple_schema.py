from dataclasses import dataclass

@dataclass
class TextNode:
    text: str
    id_: str
    metadata: dict | None = None
    @property
    def node_id(self) -> str:
        return self.id_

@dataclass
class NodeWithScore:
    node: TextNode
    score: float = 0.0
    def get_score(self) -> float:
        return self.score
