from pydantic import BaseModel
from typing import Any


class NodePosition(BaseModel):
    x: float
    y: float


class GraphNode(BaseModel):
    id: str
    type: str
    position: NodePosition
    params: dict[str, Any] = {}


class GraphEdge(BaseModel):
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class Graph(BaseModel):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    # Graph-global training config (loss/optimizer/hyperparams). Empty = defaults.
    training: dict[str, Any] = {}
    # Data-pipeline config (source, batching) driving the Data panel. Empty = defaults.
    data: dict[str, Any] = {}
