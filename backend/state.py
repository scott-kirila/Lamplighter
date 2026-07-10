"""In-memory cache of the latest design received from the editor.

The frontend pushes on every structural change, so the backend always holds the
current design. Notebook clients read it back via the HTTP API instead of
juggling exported files.

The design is a :class:`Project` (one or more models + shared training/data).
The classic single-model REST endpoints still read a :class:`Graph` view via
``get_graph`` (the schema adapter); the Project is the source of truth.
"""
from . import persist
from .schema import Graph, Project, graph_from_project

_current: Project | None = None


def set_project(project: Project) -> None:
    global _current
    _current = project
    persist.save(project)  # write-through; a no-op unless a session enabled autosave


def get_project() -> Project | None:
    return _current


def get_graph() -> Graph | None:
    """Single-model compat view of the current project (None if nothing cached)."""
    if _current is None:
        return None
    return graph_from_project(_current)
