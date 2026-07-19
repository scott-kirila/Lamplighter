"""In-memory cache of the latest design received from the editor.

The frontend pushes on every structural change, so the backend always holds the
current design. Notebook clients read it back via the HTTP API instead of
juggling exported files.

The design is a :class:`Project` (one or more models + shared training/data) —
the source of truth end to end. Endpoints that want the classic single-model view
read ``get_project().models[0]`` directly.
"""
from . import persist
from .schema import Project

_current: Project | None = None


def set_project(project: Project) -> None:
    global _current
    _current = project
    persist.save(project)  # write-through; a no-op unless a session enabled autosave


def get_project() -> Project | None:
    return _current
