import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import state
from .codegen import generate_dataloader, generate_module, generate_training
from .registry import DATA_PARAMS, REGISTRY, TRAINING_PARAMS, available_devices
from .schema import Graph, Project
from .ws import handle_ws

app = FastAPI(title="Lamplighter")


@app.get("/api/registry")
def get_registry() -> dict:
    # `emit` is backend-only codegen/inference detail — strip it so the API
    # payload (and the frontend) is unchanged by the declarative refactor.
    out = {}
    for key, node_def in REGISTRY.items():
        d = dataclasses.asdict(node_def)
        d.pop("emit", None)
        out[key] = d
    return out


@app.post("/api/codegen")
def codegen_endpoint(graph: Graph, name: str | None = None) -> dict:
    """Generate one model's module source (Export). ``name`` (a model's display
    name, sent when a project has several models) becomes the sanitized class
    name; omitted, the class is the classic ``GeneratedModel``."""
    from .codegen import sanitize_class_name

    class_name = sanitize_class_name(name) if name else "GeneratedModel"
    try:
        return {"code": generate_module(graph, class_name=class_name)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/graph")
def get_current_graph() -> dict:
    graph = state.get_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="no graph yet — open the editor first")
    return graph.model_dump()


@app.get("/api/project")
def get_current_project() -> dict:
    """The whole cached project (all models + links + shared config) — the
    editor hydrates from this so multi-model designs come back intact."""
    project = state.get_project()
    if project is None:
        raise HTTPException(status_code=404, detail="no project yet — open the editor first")
    return project.model_dump()


@app.get("/api/model/code")
def get_model_code() -> dict:
    """Codegen for the live editor graph — used by the notebook client."""
    graph = state.get_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="no graph yet — open the editor first")
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


@app.get("/api/training/params")
def get_training_params() -> list[dict]:
    """The training config form definition (rendered by the same param controls).
    The device param's choices are resolved live from the running kernel's torch,
    so only devices that actually work here are offered."""
    devices = available_devices()
    return [_param_dict(p, devices) for p in TRAINING_PARAMS]


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
    graph = state.get_graph() or Graph()
    return {"code": generate_training(graph)}


@app.post("/api/training/code")
def post_training_code(graph: Graph) -> dict:
    """Generated train() for the *posted* graph — used by the Training code panel
    so the preview matches the live editor (data-owned batch/val values and the
    model's input count included) without depending on state-sync timing."""
    return {"code": generate_training(graph)}


@app.get("/api/data/params")
def get_data_params() -> list[dict]:
    """The Data panel form definition (source, batching), rendered by the same
    param controls. `show_if` gates source-specific fields in the form."""
    return [dataclasses.asdict(p) for p in DATA_PARAMS]


@app.get("/api/data/code")
def get_data_code() -> dict:
    """Generated make_dataloaders() for the cached graph (defaults if none)."""
    graph = state.get_graph() or Graph()
    return {"code": generate_dataloader(graph)}


@app.post("/api/data/code")
def post_data_code(graph: Graph) -> dict:
    """Generated make_dataloaders() for the *posted* graph — used by the Data tab
    so the preview reflects the live editor graph (input count included) without
    depending on backend-state sync timing."""
    return {"code": generate_dataloader(graph)}


@app.post("/api/data/diagnose")
def data_diagnose(body: dict) -> dict:
    """Pre-run data↔model checks for the posted design against the session's
    registered data — shapes, dtypes, sample counts, loss/target fit, batching
    sanity. Accepts a single graph or a whole project (a multi-model recipe's
    data-fed model is checked, honoring its contract). Rendered as the Data
    tab's diagnostics checklist."""
    from .diagnose import diagnose

    design = Project.model_validate(body) if "models" in body else Graph(**body)
    return {"checks": diagnose(design)}


@app.post("/api/run/start")
def run_start(body: dict) -> dict:
    """Start an in-kernel training run. The body is a single graph (one model,
    the classic path) or a whole project (multiple models + a recipe, e.g. a
    GAN). The runner executes the same generated sources the preview panes show;
    progress streams to open tabs over the WebSocket."""
    from .runner import run_manager

    design = Project.model_validate(body) if "models" in body else Graph(**body)
    error = run_manager.start(design)
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
    """Store the last run's checkpoint under a name (overwrites an existing
    entry of the same name). 400 without a trained model."""
    from .checkpoints import save

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
    error = run_manager.restore(checkpoint)
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    return run_manager.status()


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


# Serve the built Vite bundle when running in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
