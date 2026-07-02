import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

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


@app.post("/api/run/stop")
def run_stop() -> dict:
    """Request a cooperative stop (honored at the next epoch boundary)."""
    from .runner import run_manager

    run_manager.stop()
    return {"ok": True}


@app.get("/api/run/status")
def run_status() -> dict:
    """Current run state incl. full history — serves late-joining tabs and the
    notebook client."""
    from .runner import run_manager

    return run_manager.status()


@app.get("/api/data/variables")
def get_data_variables() -> dict:
    """Live data-like variables in the notebook (tensors/arrays/Datasets/
    DataLoaders), each enriched with the Input shape it implies so the Data panel
    can offer a picker and push a shape into the model's Input node. Empty when
    there's no kernel."""
    from .introspect import input_shape_for, list_data_variables

    variables = list_data_variables()
    for v in variables:
        shape = input_shape_for(v["name"])
        if shape is not None:
            v["input_shape"] = shape
    return {"variables": variables}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await handle_ws(websocket)


# Serve the built Vite bundle when running in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
