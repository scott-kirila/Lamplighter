import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from . import state
from .codegen import generate_module
from .inference import graph_issues, infer_shapes, pin_shapes, primary_shapes
from .schema import Graph

_executor = ThreadPoolExecutor(max_workers=2)


def _validate(
    graph: Graph, want_code: bool
) -> tuple[dict, dict, dict, dict, list[str], str | None]:
    """Shape inference, graph-level issues, and — only when a tab has the code
    panel open (``want_code``) and the graph is clean — the generated module
    source. Codegen is skipped entirely when no one is watching, so a collapsed
    panel costs nothing. Code is None while anything is wrong, so the editor shows
    a placeholder instead of stale code.
    """
    params: dict[str, dict] = {}
    shapes, errors = infer_shapes(graph, param_counts=params)
    issues = graph_issues(graph)
    code: str | None = None
    if want_code and not errors and not issues:
        try:
            code = generate_module(graph)
        except ValueError:
            code = None
    # Per-node primary shape for the canvas footer; per-pin map for the Inspector
    # (so a multi-output node shows every pin's shape); per-node param counts.
    return primary_shapes(graph, shapes), pin_shapes(shapes), params, errors, issues, code


class ConnectionManager:
    """Tracks live editor connections so graph changes can be mirrored to all tabs."""

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        # Connections with the code-preview panel open. Codegen runs only while
        # this is non-empty, so a graph change costs nothing when every panel is
        # collapsed.
        self.wants_code: set[WebSocket] = set()
        # The server's event loop, captured once a connection exists, so the
        # notebook (running on a different thread) can schedule a broadcast.
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)
        self.loop = asyncio.get_running_loop()

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)
        self.wants_code.discard(websocket)

    async def broadcast(self, message: dict, exclude: WebSocket | None = None) -> None:
        for websocket in list(self.active):
            if websocket is exclude:
                continue
            try:
                await websocket.send_json(message)
            except Exception:
                # Drop dead peers; their own receive loop will clean up.
                self.active.discard(websocket)


    def broadcast_threadsafe(self, message: dict) -> None:
        """Fire-and-forget broadcast from a non-async thread (e.g. the training
        runner). Safe no-op when no client/loop exists; never blocks or raises
        into the caller — a slow socket must not throttle a training loop."""
        if self.loop is None or not self.active:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
        except Exception:
            pass

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
        # The panel starts closed on a fresh tab; it asks for code via
        # "code_preview" once opened, so skip codegen here.
        shapes, pins, params, errors, issues, code = await loop.run_in_executor(
            _executor, _validate, cached, False
        )
        await websocket.send_json({
            "type": "sync",
            "graph": cached.model_dump(),
            "shapes": shapes,
            "pin_shapes": pins,
            "params": params,
            "errors": errors,
            "graph_issues": issues,
            "code": code,
        })
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "validate":
                    graph = Graph(**msg["graph"])
                    state.set_graph(graph)
                    # Generate code if any open tab is watching — including other
                    # tabs, so an edit here keeps their preview in sync.
                    want_code = bool(manager.wants_code)
                    shapes, pins, params, errors, issues, code = await loop.run_in_executor(
                        _executor, _validate, graph, want_code
                    )
                    # Reply to the editor that made the change.
                    await websocket.send_json({
                        "type": "shapes",
                        "shapes": shapes,
                        "pin_shapes": pins,
                        "params": params,
                        "errors": errors,
                        "graph_issues": issues,
                        "code": code,
                    })
                    # Mirror the new graph to every other open tab.
                    await manager.broadcast(
                        {
                            "type": "sync",
                            "graph": graph.model_dump(),
                            "shapes": shapes,
                            "pin_shapes": pins,
                            "params": params,
                            "errors": errors,
                            "graph_issues": issues,
                            "code": code,
                        },
                        exclude=websocket,
                    )
                elif msg.get("type") == "code_preview":
                    # A tab opened/closed its code panel. While open it joins the
                    # watcher set; on open it also gets the current code pushed so
                    # the panel fills immediately without waiting for an edit.
                    if msg.get("enabled"):
                        manager.wants_code.add(websocket)
                        cached = state.get_graph()
                        code = None
                        if cached is not None:
                            _, _, _, _, _, code = await loop.run_in_executor(
                                _executor, _validate, cached, True
                            )
                        await websocket.send_json({"type": "code", "code": code})
                    else:
                        manager.wants_code.discard(websocket)
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
