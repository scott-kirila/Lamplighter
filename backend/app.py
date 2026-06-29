import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

from . import state
from .codegen import generate_module, generate_training
from .registry import REGISTRY, TRAINING_PARAMS
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
    """The training config form definition (rendered by the same param controls)."""
    return [dataclasses.asdict(p) for p in TRAINING_PARAMS]


@app.get("/api/training/code")
def get_training_code() -> dict:
    """Generated train() function for the current config (defaults if no graph)."""
    graph = state.get_graph() or Graph()
    return {"code": generate_training(graph)}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await handle_ws(websocket)


# Serve the built Vite bundle when running in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
