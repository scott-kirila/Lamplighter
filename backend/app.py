import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import state
from .codegen import generate_dataloader, generate_module, generate_training
from .registry import DATA_PARAMS, REGISTRY, TRAINING_PARAMS, available_devices
from .schema import Graph
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
def codegen_endpoint(graph: Graph) -> dict:
    try:
        return {"code": generate_module(graph)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/graph")
def get_current_graph() -> dict:
    graph = state.get_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="no graph yet — open the editor first")
    return graph.model_dump()


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


@app.get("/api/training/params")
def get_training_params() -> list[dict]:
    """The training config form definition (rendered by the same param controls).
    The device param's choices are resolved live from the running kernel's torch,
    so only devices that actually work here are offered."""
    devices = available_devices()
    out: list[dict] = []
    for p in TRAINING_PARAMS:
        d = dataclasses.asdict(p)
        if p.name == "device":
            d["choices"] = devices
        out.append(d)
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
def data_diagnose(graph: Graph) -> dict:
    """Pre-run data↔model checks for the posted (live editor) graph against the
    session's registered data — shapes, dtypes, sample counts, loss/target fit,
    batching sanity. Rendered as the Data tab's diagnostics checklist."""
    from .diagnose import diagnose

    return {"checks": diagnose(graph)}


@app.post("/api/run/start")
def run_start(graph: Graph) -> dict:
    """Start an in-kernel training run for the posted (live editor) graph. The
    runner executes the same generated sources the preview panes show; progress
    streams to open tabs over the WebSocket."""
    from .runner import run_manager

    error = run_manager.start(graph)
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
