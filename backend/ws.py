import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from . import state
from .inference import graph_issues, infer_shapes
from .schema import Graph

_executor = ThreadPoolExecutor(max_workers=2)


def _validate(graph: Graph) -> tuple[dict, dict, list[str]]:
    """Shape inference plus graph-level issues for one graph."""
    shapes, errors = infer_shapes(graph)
    return shapes, errors, graph_issues(graph)


class ConnectionManager:
    """Tracks live editor connections so graph changes can be mirrored to all tabs."""

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        # The server's event loop, captured once a connection exists, so the
        # notebook (running on a different thread) can schedule a broadcast.
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)
        self.loop = asyncio.get_running_loop()

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict, exclude: WebSocket | None = None) -> None:
        for websocket in list(self.active):
            if websocket is exclude:
                continue
            try:
                await websocket.send_json(message)
            except Exception:
                # Drop dead peers; their own receive loop will clean up.
                self.active.discard(websocket)


    def notify_stopped(self, timeout: float = 2.0) -> None:
        """Tell every open editor the session is ending, from another thread.

        Called by the notebook's ``stop()`` before the server is torn down, so
        tabs can show a "session stopped" state instead of silently retrying.
        Blocks briefly until the messages are flushed.
        """
        if self.loop is None or not self.active:
            return
        future = asyncio.run_coroutine_threadsafe(
            self.broadcast({"type": "session_stopped"}), self.loop
        )
        try:
            future.result(timeout=timeout)
        except Exception:
            pass


manager = ConnectionManager()


async def handle_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    loop = asyncio.get_running_loop()
    # Hand the new tab the current design up front, so a late joiner or a tab
    # reconnecting after a restart rehydrates instead of sitting blank — and
    # never has to push its own (possibly empty) canvas just to find out.
    cached = state.get_graph()
    if cached is not None:
        shapes, errors, issues = await loop.run_in_executor(_executor, _validate, cached)
        await websocket.send_json({
            "type": "sync",
            "graph": cached.model_dump(),
            "shapes": shapes,
            "errors": errors,
            "graph_issues": issues,
        })
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "validate":
                    graph = Graph(**msg["graph"])
                    state.set_graph(graph)
                    shapes, errors, issues = await loop.run_in_executor(
                        _executor, _validate, graph
                    )
                    # Reply to the editor that made the change.
                    await websocket.send_json({
                        "type": "shapes",
                        "shapes": shapes,
                        "errors": errors,
                        "graph_issues": issues,
                    })
                    # Mirror the new graph to every other open tab.
                    await manager.broadcast(
                        {
                            "type": "sync",
                            "graph": graph.model_dump(),
                            "shapes": shapes,
                            "errors": errors,
                            "graph_issues": issues,
                        },
                        exclude=websocket,
                    )
                elif msg.get("type") == "moves":
                    # Drag-end position update: patch the cache, mirror to others,
                    # no shape inference (positions don't affect shapes).
                    moves = msg.get("nodes", [])
                    cached = state.get_graph()
                    if cached is not None:
                        pos_by_id = {m["id"]: m["position"] for m in moves}
                        for node in cached.nodes:
                            new_pos = pos_by_id.get(node.id)
                            if new_pos is not None:
                                node.position.x = new_pos["x"]
                                node.position.y = new_pos["y"]
                    await manager.broadcast(
                        {"type": "moves", "nodes": moves}, exclude=websocket
                    )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
