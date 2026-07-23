"""Install an imported model into the current project.

The seam between :mod:`importer` (fx trace → graph, a pure function of a model)
and the session state (the project the app edits, the datastore the runner
reads). Kept out of importer.py so that module stays a pure graph builder with
no knowledge of sessions.
"""
from __future__ import annotations

from typing import Any

from . import datastore, state
from .importer import ImportError_, trace
from .schema import ImportInfo, ModelDef, Project


def _resolve_shape(model, x_or_shape) -> tuple[int, ...]:
    """The input shape to trace with. A real tensor is the primary form — the
    user usually has a batch already — with an explicit shape as the fallback.
    A tensor's shape is used verbatim (it already has a batch dim)."""
    import torch

    if isinstance(x_or_shape, torch.Tensor):
        return tuple(x_or_shape.shape)
    if x_or_shape is None:
        raise ImportError_(
            "inspect() needs to know the input shape — pass a real example batch "
            "(sess.inspect(model, x)) or an explicit shape "
            "(sess.inspect(model, input_shape=(1, 3, 224, 224)))."
        )
    return tuple(int(d) for d in x_or_shape)


def inspect_model(model, x_or_shape, name: str | None = None) -> dict[str, Any]:
    """Trace ``model``, and — unless it was refused — install it as a new model
    in the current project and stash its weights for the runner. Returns a
    notebook-facing report (node/opaque counts, fidelity findings, whether it's
    runnable).

    Refused or holey models are reported but NOT installed: the canvas should
    never gain a model you can't run, silently.
    """
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise ImportError_(
            f"inspect() takes an nn.Module, got {type(model).__name__}"
        )

    shape = _resolve_shape(model, x_or_shape)
    was_training = model.training
    result = trace(model.eval(), shape)
    if was_training:
        model.train()  # leave the caller's model exactly as we found it

    report: dict[str, Any] = {
        "source": type(model).__name__,
        "input_shape": list(shape),
        "nodes": len(result["graph"].nodes),
        "opaque": result["opaque_count"],
        "refused": result["refused"],
        "refused_reason": result["refused_reason"],
        "findings": result["findings"],
        "runnable": result["opaque_count"] == 0 and not result["refused"],
        "installed": False,
    }
    if result["refused"]:
        return report

    model_id = _install(model, result, name)
    report["installed"] = True
    report["model_id"] = model_id
    report["model_name"] = _project().models[-1].name
    return report


def _project() -> Project:
    project = state.get_project()
    return project if project is not None else Project()


def _install(model, result, name: str | None) -> str:
    """Add the traced graph as a new ModelDef and stash its weights. A single
    empty scaffold is REPLACED (the common case — a fresh session); otherwise
    the model is appended, so importing several composes them on the overview."""
    from .importer import trace as _  # noqa: F401  (keep import local + explicit)

    project = _project()
    source = type(model).__name__
    model_name = name or _unique_name(project, source)
    model_id = _unique_id(project, model_name)

    md = ModelDef(
        id=model_id,
        name=model_name,
        graph=result["graph"],
        imported=ImportInfo(source=source, state_keys=list(result["state_keys"])),
    )

    # Replace a lone untouched scaffold (Input→Output, nothing wired) rather
    # than leaving it beside the import — the fresh-session case.
    if _is_scaffold(project):
        project.models = [md]
    else:
        project.models = [*project.models, md]
    state.set_project(project)

    datastore.register_import(model_id, list(model.state_dict().values()))
    return model_id


def _is_scaffold(project: Project) -> bool:
    if len(project.models) != 1 or project.data_nodes or project.links:
        return False
    g = project.models[0].graph
    if g.edges or len(g.nodes) > 2:
        return False
    return all(n.type in ("Input", "Output") for n in g.nodes)


def _unique_name(project: Project, base: str) -> str:
    existing = {m.name for m in project.models}
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def _unique_id(project: Project, name: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
    existing = {m.id for m in project.models}
    if slug not in existing:
        return slug
    i = 2
    while f"{slug}-{i}" in existing:
        i += 1
    return f"{slug}-{i}"
