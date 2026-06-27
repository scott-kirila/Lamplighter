import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

from .codegen import generate_module
from .registry import REGISTRY
from .schema import Graph
from .ws import handle_ws

app = FastAPI(title="Scorch")


@app.get("/api/registry")
def get_registry() -> dict:
    return {key: dataclasses.asdict(node_def) for key, node_def in REGISTRY.items()}


@app.post("/api/codegen")
def codegen_endpoint(graph: Graph) -> dict:
    try:
        return {"code": generate_module(graph)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await handle_ws(websocket)


# Serve the built Vite bundle when running in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
