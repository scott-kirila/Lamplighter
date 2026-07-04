"""In-memory cache of the latest design received from the editor.

The frontend pushes on every structural change, so the backend always holds the
current design. Notebook clients read it back via the HTTP API instead of
juggling exported files.

The design is a :class:`Project` (one or more models + shared training/data).
Single-model callers still speak :class:`Graph` through ``get_graph`` /
``set_graph``, which convert via the schema adapters — so the classic path is
unchanged while the Project becomes the source of truth underneath.
"""
from . import persist
from .schema import Graph, Project, graph_from_project, project_from_graph

_current: Project | None = None


def set_project(project: Project) -> None:
    global _current
    _current = project
    persist.save(project)  # write-through; a no-op unless a session enabled autosave


def get_project() -> Project | None:
    return _current


def set_graph(graph: Graph) -> None:
    """Single-model compat: wrap the Graph as a one-model project and store it."""
    set_project(project_from_graph(graph))


def get_graph() -> Graph | None:
    """Single-model compat view of the current project (None if nothing cached)."""
    if _current is None:
        return None
    return graph_from_project(_current)
