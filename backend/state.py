"""In-memory cache of the latest graph received from the editor.

The frontend pushes the graph over the WebSocket on every structural change,
so the backend always holds the current design. Notebook clients read it back
via the HTTP API instead of juggling exported files.
"""
from .schema import Graph

_current: Graph | None = None


def set_graph(graph: Graph) -> None:
    global _current
    _current = graph


def get_graph() -> Graph | None:
    return _current
