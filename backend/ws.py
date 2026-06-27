import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from . import state
from .inference import infer_shapes
from .schema import Graph

_executor = ThreadPoolExecutor(max_workers=2)


async def handle_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "validate":
                    graph = Graph(**msg["graph"])
                    state.set_graph(graph)
                    shapes, errors = await loop.run_in_executor(
                        _executor, infer_shapes, graph
                    )
                    await websocket.send_json({
                        "type": "shapes",
                        "shapes": shapes,
                        "errors": errors,
                    })
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                # Report per-message failures without tearing down the socket
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
