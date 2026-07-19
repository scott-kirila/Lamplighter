"""The FastAPI app — HTTP + WebSocket surface over the in-kernel session state.

Routes cover the registry, code generation, shape inference, run lifecycle,
checkpoints, templates, and project persistence, and serve the built frontend.
Handlers stay thin: each delegates to a backend module (codegen, inference,
runner, checkpoints, …), so this file is wiring, not logic.
"""
import dataclasses

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import state
from .codegen import generate_dataloader, generate_module, generate_training
from .registry import DATA_PARAMS, REGISTRY, available_devices
from .schema import Graph, Project, resolve_data_config
from .ws import handle_ws

app = FastAPI(title="Lamplighter")


def _single_model_view() -> tuple[Graph, dict, dict]:
    """The classic single-model view of the cached project — its first model's
    graph plus the project-level training config and the data config wired into
    that model. What the notebook client and the preview endpoints read now that
    training/data live on the Project, not the Graph. Empty defaults when nothing
    is cached yet."""
    project = state.get_project()
    if project is None or not project.models:
        return Graph(), {}, {}
    model = project.models[0]
    return model.graph, dict(project.training or {}), resolve_data_config(project, model.id)


@app.get("/api/registry")
def get_registry() -> dict:
    # `emit` is backend-only codegen/inference detail — strip it so the API
    # payload (and the frontend) is unchanged by the declarative refactor.
    # `doc` is enriched into {summary, body}: live torch docstrings for
    # nn-backed nodes, the authored one-liner for Lamplighter-native ones.
    from .registry import node_doc

    out = {}
    for key, node_def in REGISTRY.items():
        d = dataclasses.asdict(node_def)
        d.pop("emit", None)
        d["doc"] = node_doc(node_def)
        out[key] = d
    return out


@app.get("/api/graph")
def get_current_graph() -> dict:
    """The current editor graph as JSON (nodes + edges) — the first model of the
    cached project, for the notebook client's ``lamplighter.graph()``."""
    project = state.get_project()
    if project is None:
        raise HTTPException(status_code=404, detail="no graph yet — open the editor first")
    graph = project.models[0].graph if project.models else Graph()
    return graph.model_dump()


@app.get("/api/project")
def get_current_project() -> dict:
    """The whole cached project (all models + links + shared config) — the
    editor hydrates from this so multi-model projects come back intact."""
    project = state.get_project()
    if project is None:
        raise HTTPException(status_code=404, detail="no project yet — open the editor first")
    return project.model_dump()


@app.get("/api/model/code")
def get_model_code() -> dict:
    """Codegen for the live editor graph — used by the notebook client."""
    project = state.get_project()
    if project is None:
        raise HTTPException(status_code=404, detail="no graph yet — open the editor first")
    graph = project.models[0].graph if project.models else Graph()
    try:
        return {"code": generate_module(graph)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _param_dict(p, devices: list[str]) -> dict:
    """A ParamDef as the form consumes it, with the device param's choices
    resolved live from the running kernel's torch."""
    d = dataclasses.asdict(p)
    if p.name == "device":
        d["choices"] = devices
    return d


@app.get("/api/templates")
def list_templates() -> dict:
    """The built-in starting points for the New-project flow (metadata only —
    the picker's rows). Each template is a complete working project, held
    green by the test suite."""
    from .templates import TEMPLATES

    return {
        "templates": [
            {"name": t.name, "label": t.label, "description": t.description}
            for t in TEMPLATES.values()
        ]
    }


@app.get("/api/templates/{name}")
def get_template(name: str) -> dict:
    """One template's full project, ready for the editor's loadProject path."""
    from .templates import TEMPLATES

    t = TEMPLATES.get(name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"no template named '{name}'")
    return t.build().model_dump()


@app.get("/api/recipes")
def get_recipes() -> list[dict]:
    """The training recipe definitions — loop templates the Training tab renders
    (roles, loop params, per-role params) and the runner generates from. The
    generator function is backend-only (like a node's `emit`), so it's stripped
    from the payload; `needs_targets`/`has_val` carry each recipe's data
    contract."""
    from .recipes import RECIPES

    devices = available_devices()
    out: list[dict] = []
    for r in RECIPES.values():
        out.append({
            "name": r.name,
            "label": r.label,
            "roles": [dataclasses.asdict(role) for role in r.roles],
            "params": [_param_dict(p, devices) for p in r.params],
            "role_params": {
                role: [_param_dict(p, devices) for p in params]
                for role, params in r.role_params.items()
            },
            "needs_targets": r.needs_targets,
            "has_val": r.has_val,
            "data_role": r.data_role,
        })
    return out


@app.get("/api/training/code")
def get_training_code() -> dict:
    """Generated train() function for the current config (defaults if no graph)."""
    graph, training, _ = _single_model_view()
    return {"code": generate_training(graph, training)}


@app.post("/api/training/code")
def post_training_code(project: Project) -> dict:
    """Generated train() for the *posted* project — used by the Training code panel
    so the preview matches the live editor (the model's input count and the
    project's training config) without depending on state-sync timing."""
    graph = project.models[0].graph if project.models else Graph()
    return {"code": generate_training(graph, project.training)}


@app.get("/api/data/params")
def get_data_params() -> list[dict]:
    """The Data panel form definition (source, batching), rendered by the same
    param controls. `show_if` gates source-specific fields in the form."""
    return [dataclasses.asdict(p) for p in DATA_PARAMS]


@app.get("/api/data/code")
def get_data_code() -> dict:
    """Generated make_dataloaders() for the cached project's data (defaults if none)."""
    graph, _, data = _single_model_view()
    return {"code": generate_dataloader(graph, data)}


@app.post("/api/data/code")
def post_data_code(project: Project) -> dict:
    """Generated make_dataloaders() for the *posted* project — the first model's
    graph (input count) and the data config wired into it."""
    graph = project.models[0].graph if project.models else Graph()
    data = resolve_data_config(project, project.models[0].id) if project.models else {}
    return {"code": generate_dataloader(graph, data)}


@app.post("/api/data/diagnose")
def data_diagnose(body: dict) -> dict:
    """Pre-run data↔model checks for the posted project against the session's
    registered data — shapes, dtypes, sample counts, loss/target fit, batching
    sanity. A multi-model recipe's data-fed model is checked, honoring its
    contract. Rendered as the Data tab's diagnostics checklist."""
    from .diagnose import diagnose

    return {"checks": diagnose(Project.model_validate(body))}


@app.post("/api/run/start")
def run_start(body: dict) -> dict:
    """Start an in-kernel training run. The body is a whole project (one or more
    models + a recipe — a GAN sends several). The runner executes the same
    generated sources the preview panes show; progress streams to open tabs over
    the WebSocket."""
    from .runner import run_manager

    error = run_manager.start(Project.model_validate(body))
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    return {"ok": True}


class ResumeRequest(BaseModel):
    name: str
    epochs: int | None = None


@app.post("/api/run/resume")
def run_resume(body: ResumeRequest) -> dict:
    """Warm-start from a stored checkpoint, continuing toward its planned epoch
    target — `epochs` (a TOTAL, like everywhere else) raises the target for a
    finished run; omitted, an interrupted run finishes its plan. The
    checkpoint's OWN graph/config/sources run (not the live canvas), final
    weights loaded, fresh optimizer, newly drawn seed; epoch numbering
    continues. Returns the starting status (with the preloaded history) so the
    acting tab can seed its charts."""
    from .checkpoints import load
    from .runner import run_manager

    try:
        checkpoint = load(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    error = run_manager.resume(body.name, checkpoint, epochs=body.epochs)
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    return run_manager.status()


@app.post("/api/run/stop")
def run_stop() -> dict:
    """Request a cooperative stop (honored at the next epoch boundary)."""
    from .runner import run_manager

    run_manager.stop()
    return {"ok": True}


@app.get("/api/run/weights")
def run_weights():
    """Download the last run's checkpoint (weights + snapshot) as model.pt —
    the same self-contained format sess.save_checkpoint() writes, loadable via
    lamplighter.load_checkpoint()."""
    import io

    import torch
    from fastapi import Response

    from .runner import run_manager

    try:
        checkpoint = run_manager.checkpoint()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="model.pt"'},
    )


@app.get("/api/run/status")
def run_status() -> dict:
    """Current run state incl. full history — serves late-joining tabs and the
    notebook client."""
    from .runner import run_manager

    return run_manager.status()


@app.get("/api/run/preview")
def run_preview(role: str | None = None, n: int = 16) -> dict:
    """A sample of the trained model's input → output on real data (resolved from
    each input's wired data node). Generic — the frontend renders each tensor by
    its shape. {"error": ...} when a preview isn't available (no run, data gone)."""
    from .runner import run_manager

    return run_manager.preview(role=role, n=n)


class CheckpointName(BaseModel):
    name: str


@app.get("/api/checkpoints")
def list_checkpoints() -> dict:
    """The session's stored checkpoints (metadata only). Mutations are also
    pushed live over the WS; this is the pull path (initial load)."""
    from .checkpoints import metas

    return {"checkpoints": metas()}


@app.post("/api/checkpoints")
def save_checkpoint_endpoint(body: CheckpointName) -> dict:
    """Store the live model's weights under ``name`` (overwrites a same-named
    entry). This clones whatever the kernel currently holds, so it must not
    overwrite a DIFFERENT run's auto record: after a restore (or a newer run)
    the live model belongs to another run, and keeping it under an old run-N
    slot would mislabel those weights. Refuse that one case with 409; keeping
    the kernel's own current run, or saving under any other name, is fine. 400
    without a trained model. (Notebook saves call checkpoints.save() directly,
    so they stay free to name an arbitrary snapshot.)"""
    from .checkpoints import is_auto, save
    from .runner import run_manager

    if body.name != run_manager.run_name and is_auto(body.name):
        raise HTTPException(
            status_code=409,
            detail="the kernel no longer holds this run's weights — a restore "
            "or a newer run replaced them",
        )
    try:
        return {"checkpoint": save(body.name)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/checkpoints/{name}")
def delete_checkpoint_endpoint(name: str) -> dict:
    from .checkpoints import delete

    try:
        delete(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@app.post("/api/checkpoints/{name}/rename")
def rename_run_endpoint(name: str, body: CheckpointName) -> dict:
    """Rename a stored run. Naming is keep intent — the entry leaves the
    auto-record retention pool."""
    from .checkpoints import rename

    try:
        return {"checkpoint": rename(name, body.name)}
    except ValueError as exc:
        code = 404 if "no run named" in str(exc) else 409
        raise HTTPException(status_code=code, detail=str(exc))


@app.get("/api/checkpoints/{name}/preview")
def preview_run_endpoint(name: str, role: str | None = None, n: int = 16) -> dict:
    """A stored run's input → output sample — rebuilds the run's model from its
    saved weights and forwards sample inputs, WITHOUT touching the kernel's live
    model (so the Preview tab can flip between runs). 409 for a weightless run
    (no saved weights to rebuild); 404 for an unknown name. Same payload shape as
    /api/run/preview."""
    from .checkpoints import load
    from .runner import run_manager

    try:
        checkpoint = load(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if checkpoint.get("state_dicts") is None:
        raise HTTPException(
            status_code=409,
            detail="this run's weights weren't saved, so it can't be previewed — "
            "＋ save weights on a run while it's the current one to preview it later",
        )
    return run_manager.preview_checkpoint(checkpoint, role=role, n=n)


@app.get("/api/checkpoints/{name}/view")
def view_run_endpoint(name: str) -> dict:
    """A stored run as a status-shaped payload — everything the dashboard
    needs to SHOW it (curves, health, steps, config, seed) without touching
    the run manager or the kernel's model. Works for weightless records;
    restore/resume stay the explicit, weights-requiring actions."""
    from .checkpoints import load
    from .runner import run_config_from

    try:
        checkpoint = load(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    snapshot = checkpoint.get("snapshot") or {}
    plan = (snapshot.get("training") or {}).get("epochs")
    return {
        "name": name,
        "state": snapshot.get("state") or "done",
        "error": None,
        "epoch": checkpoint.get("epoch"),
        "epochs": int(plan) if plan is not None else checkpoint.get("epoch"),
        "seed": snapshot.get("seed"),
        "best_epoch": checkpoint.get("best_epoch"),
        "history": checkpoint.get("history") or {},
        "health_history": checkpoint.get("health_history") or [],
        "steps": checkpoint.get("steps") or [],
        "step_total": checkpoint.get("step_total") or 0,
        "config": run_config_from(snapshot),
    }


@app.get("/api/checkpoints/{name}/history")
def checkpoint_history(name: str) -> dict:
    """A stored run's full per-epoch history plus the training config that
    produced it — what the run-comparison overlay charts and diff table read.
    (Metas stay light; this is fetched per checkpoint when compared.)"""
    from .checkpoints import load

    try:
        checkpoint = load(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    snapshot = checkpoint.get("snapshot") or {}
    return {
        "name": name,
        "history": checkpoint.get("history") or {},
        "training": snapshot.get("training") or {},
    }


@app.get("/api/checkpoints/{name}/weights")
def checkpoint_weights(name: str):
    """Download a stored checkpoint — the same self-contained format as
    /api/run/weights, loadable via lamplighter.load_checkpoint()."""
    import io

    import torch
    from fastapi import Response

    from .checkpoints import load

    try:
        checkpoint = load(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if checkpoint.get("state_dicts") is None:
        raise HTTPException(status_code=409, detail="this run kept no weights — nothing to download")
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    filename = "".join(c for c in name if c.isalnum() or c in "-_.") or "checkpoint"
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pt"'},
    )


@app.post("/api/checkpoints/{name}/restore")
def restore_checkpoint_endpoint(name: str) -> dict:
    """Repopulate the run manager from a stored checkpoint (400 while a run is
    in progress). Returns the new run status — the acting tab replaces its run
    state from it; other tabs pick it up on their next connect."""
    from .checkpoints import load
    from .runner import run_manager

    try:
        checkpoint = load(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    error = run_manager.restore(checkpoint, name=name)
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    return run_manager.status()


@app.get("/api/modules")
def get_registered_modules() -> dict:
    """The session's registered custom nn.Module classes
    (sess.modules(Name=Class) in the notebook) — the Custom node's picker."""
    from .datastore import module_summaries

    return {"modules": module_summaries()}


@app.get("/api/data/variables")
def get_data_variables() -> dict:
    """The session's registered data (sess.data(X=X, y=y) in the notebook), each
    entry enriched with the Input shape it implies — so the Data panel offers a
    small curated picker and can push a shape into the model's Input node.
    Registry changes are also pushed live over the WS; this remains the pull
    path (initial load + the ↻ refresh fallback)."""
    from .datastore import enriched_variables

    return {"variables": enriched_variables()}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await handle_ws(websocket)


# Serve the built UI — the packaged bundle (release wheels) or the checkout's
# frontend/dist (dev). Resolved at import time; the session builds the dev
# bundle BEFORE importing this module so the mount registers (see
# session._start_server's ordering note).
from .dist import frontend_dist  # noqa: E402 — after the routes, by design

_frontend_dist = frontend_dist()
if _frontend_dist is not None:
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
