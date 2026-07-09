import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from . import state
from .codegen import class_name_for, generate_module
from .inference import (
    data_node_output_shape,
    graph_issues,
    infer_shapes,
    link_issues,
    pin_shapes,
    primary_shapes,
)
from .schema import Graph, Project

_executor = ThreadPoolExecutor(max_workers=2)


def _infer_model(graph: Graph, want_code: bool, class_name: str = "GeneratedModel") -> dict:
    """Shape inference, graph-level issues, and — only when a tab has the code
    panel open (``want_code``) and the graph is clean — the generated module
    source, for ONE model's graph. Codegen is skipped when no one is watching,
    so a collapsed panel costs nothing. Code is None while anything is wrong, so
    the editor shows a placeholder instead of stale code."""
    params: dict[str, dict] = {}
    shapes, errors = infer_shapes(graph, param_counts=params)
    issues = graph_issues(graph)
    code: str | None = None
    if want_code and not errors and not issues:
        try:
            code = generate_module(graph, class_name=class_name)
        except ValueError:
            code = None
    return {
        # Per-node primary shape for the canvas footer; per-pin map for the
        # Inspector (so a multi-output node shows every pin's shape); per-node
        # param counts.
        "shapes": primary_shapes(graph, shapes),
        "pin_shapes": pin_shapes(shapes),
        "params": params,
        "errors": errors,
        "graph_issues": issues,
        "code": code,
    }


def _validate_project(project: Project, want_code: bool) -> tuple[dict, dict, list[dict]]:
    """Per-model inference for the whole project. Returns
    ``(models, code, links)`` — ``models`` is ``{model_id: {shapes, pin_shapes,
    params, errors, graph_issues}}``, ``code`` is ``{model_id: source|None}``,
    and ``links`` is the per-link shape-check result — so each tab renders its
    own active model and the system view's link evidence from one broadcast."""
    models: dict[str, dict] = {}
    code: dict[str, str | None] = {}
    sole = len(project.models) <= 1
    for m in project.models:
        result = _infer_model(m.graph, want_code, class_name=class_name_for(m.name, sole))
        code[m.id] = result.pop("code")
        models[m.id] = result
    # Resolve each data node's output shape (noise from its dims, a memory dataset
    # from the picked variable) so data→model wires can be shape-checked too.
    from .datastore import registry

    ns = registry()
    data_shapes: dict[str, dict[str, list[int]]] = {}
    for dn in project.data_nodes:
        pins = {}
        for pin in ("x", "y"):
            shape = data_node_output_shape(dn, ns, pin)
            if shape is not None:
                pins[pin] = shape
        if pins:
            data_shapes[dn.id] = pins
    links = link_issues(project, {mid: r["shapes"] for mid, r in models.items()}, data_shapes)
    return models, code, links


class ConnectionManager:
    """Tracks live editor connections so project changes can be mirrored to all tabs."""

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        # Connections with the code-preview panel open. Codegen runs only while
        # this is non-empty, so a change costs nothing when every panel is
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


def _project_from_message(msg: dict) -> Project:
    """The project a tab sent. The editor is project-native — ``validate``
    always carries ``project`` (the bare-``graph`` wire shape had no remaining
    senders and was removed)."""
    return Project.model_validate(msg["project"])


async def handle_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    loop = asyncio.get_running_loop()
    # Hand the new tab the current design up front, so a late joiner or a tab
    # reconnecting after a restart rehydrates instead of sitting blank — and
    # never has to push its own (possibly empty) canvas just to find out.
    cached = state.get_project()
    if cached is not None:
        # The panel starts closed on a fresh tab; it asks for code via
        # "code_preview" once opened, so skip codegen here.
        models, code, links = await loop.run_in_executor(
            _executor, _validate_project, cached, False
        )
        await websocket.send_json({
            "type": "sync",
            "project": cached.model_dump(),
            "models": models,
            "code": code,
            "links": links,
        })
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "validate":
                    project = _project_from_message(msg)
                    state.set_project(project)
                    # Generate code if any open tab is watching — including other
                    # tabs, so an edit here keeps their preview in sync.
                    want_code = bool(manager.wants_code)
                    models, code, links = await loop.run_in_executor(
                        _executor, _validate_project, project, want_code
                    )
                    # Reply to the editor that made the change.
                    await websocket.send_json(
                        {"type": "shapes", "models": models, "code": code, "links": links}
                    )
                    # Mirror the new project to every other open tab.
                    await manager.broadcast(
                        {
                            "type": "sync",
                            "project": project.model_dump(),
                            "models": models,
                            "code": code,
                            "links": links,
                        },
                        exclude=websocket,
                    )
                elif msg.get("type") == "code_preview":
                    # A tab opened/closed its code panel. While open it joins the
                    # watcher set; on open it also gets the current code pushed so
                    # the panel fills immediately without waiting for an edit.
                    if msg.get("enabled"):
                        manager.wants_code.add(websocket)
                        cached = state.get_project()
                        code: dict[str, str | None] = {}
                        if cached is not None:
                            _, code, _ = await loop.run_in_executor(
                                _executor, _validate_project, cached, True
                            )
                        await websocket.send_json({"type": "code", "code": code})
                    else:
                        manager.wants_code.discard(websocket)
                elif msg.get("type") == "moves":
                    # Drag-end position update within a model's canvas: patch that
                    # model's node positions, mirror to others, no shape inference.
                    model_id = msg.get("model_id")
                    moves = msg.get("nodes", [])
                    project = state.get_project()
                    if project is not None:
                        pos_by_id = {m["id"]: m["position"] for m in moves}
                        for model in project.models:
                            if model_id is not None and model.id != model_id:
                                continue
                            for node in model.graph.nodes:
                                new_pos = pos_by_id.get(node.id)
                                if new_pos is not None:
                                    node.position.x = new_pos["x"]
                                    node.position.y = new_pos["y"]
                        # Re-set the (mutated-in-place) project so the autosave
                        # write-through sees the new positions too.
                        state.set_project(project)
                    await manager.broadcast(
                        {"type": "moves", "model_id": model_id, "nodes": moves},
                        exclude=websocket,
                    )
                elif msg.get("type") == "system_moves":
                    # Drag-end on the system canvas: patch model + data-node
                    # sys_positions.
                    moves = msg.get("nodes", [])
                    project = state.get_project()
                    if project is not None:
                        pos_by_id = {m["id"]: m["position"] for m in moves}
                        for item in (*project.models, *project.data_nodes):
                            new_pos = pos_by_id.get(item.id)
                            if new_pos is not None:
                                item.sys_position.x = new_pos["x"]
                                item.sys_position.y = new_pos["y"]
                        state.set_project(project)
                    await manager.broadcast(
                        {"type": "system_moves", "nodes": moves}, exclude=websocket
                    )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
