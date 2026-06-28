# Lamplighter

[![CI](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml/badge.svg)](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml)

A visual PyTorch model builder. You assemble an `nn.Module` by dragging nodes
onto a canvas and connecting them, and Lamplighter infers the tensor shape at
each step as you go. When the graph is ready, you load it into a notebook as a
real `torch.nn.Module` — the generated model tracks whatever is currently on the
canvas, so there are no files to export and re-import by hand.

## Architecture

Lamplighter is three parts that run locally:

- **Backend** (`backend/`) — a FastAPI app. It holds the current graph in
  memory, infers tensor shapes, generates `nn.Module` source from the graph, and
  serves the built frontend. API and editor share a single port.
- **Frontend** (`frontend/`) — a React + [xyflow](https://reactflow.dev) editor:
  node palette, canvas, and an inspector for node parameters.
- **Client** (`lamplighter/`) — the notebook API and the session lifecycle that starts/stops the server from inside the kernel.

The graph is cached in the backend rather than in any one browser tab, and
changes are synced to every open tab over a WebSocket. So you can edit from
several tabs at once, and closing and reopening a tab restores your work. The
whole session is driven from a Jupyter cell rather than a separate terminal.

## Usage

Run the notebook from the project root so `import lamplighter` resolves.

```python
import lamplighter

sess = lamplighter.start()        # serve the app, open the editor
# build a model on the canvas...

model = lamplighter.build_model() # nn.Module from the current canvas
print(lamplighter.model_code())   # generated source

lamplighter.open_editor()         # reopen the tab (work is restored)
lamplighter.stop()                # stop the session
```

`build_model()` reads the live graph each time it's called, so re-run it after
editing the canvas to pick up your changes. A full walkthrough is in
[`example.ipynb`](example.ipynb).

### Building a model

Drag nodes from the palette onto the canvas and connect their pins. Every graph
starts at an **Input** node — set its shape as comma-separated dimensions, e.g.
`1, 784` for `(batch, features)` or `1, 3, 28, 28` for `(batch, channels, H, W)`
— and ends at an **Output** node. As you wire nodes together the shape at each
pin is inferred and shown, and connections that don't typecheck are flagged. To
get the code out, use the editor's **Export** button or `lamplighter.model_code()`.

Available nodes:

| Category    | Nodes                                          |
|-------------|------------------------------------------------|
| I/O         | Input, Output                                  |
| Layers      | Linear, Conv2d, Flatten, Dropout, BatchNorm1d |
| Activations | ReLU, Sigmoid, Tanh                            |
| Ops         | Concat                                         |

## API

| Call | Description |
|------|-------------|
| `start(port=8000, host="127.0.0.1", open_browser=True, build="auto")` | Start a new session, or reuse the running one. Returns a `Session`. |
| `stop()` | Stop the current session. Open tabs show a "session stopped" overlay. |
| `open_editor()` | Reopen the editor tab for the running session. |
| `build_model()` | Instantiate the current graph as a `torch.nn.Module`. |
| `model_code()` | The generated module source, as a string. |
| `graph()` | The current graph (nodes + edges) as JSON. |
| `status()` | `{running, url, has_graph}` for the current session. |
| `current()` | The current `Session`, or `None`. |

The `build` argument controls the frontend build: `"auto"` builds only if
`dist/` is missing, `True` always rebuilds, and `False` never builds (and fails
if `dist/` is absent).

## Development

The notebook flow builds the frontend for you. To work on the project directly:

```bash
uv sync                          # Python dependencies

# Backend (standalone, with autoreload)
uv run python main.py            # http://127.0.0.1:8000

# Frontend
cd frontend
npm install
npm run build                    # produces dist/, which the backend serves
npm run dev                      # or: Vite dev server with hot-reload
```

The backend serves the built `dist/` on each request, so after editing frontend
source you need to run `npm run build` and hard-refresh the browser to see the
change. `lamplighter.start()` only auto-builds when `dist/` is missing, so it
won't rebuild on source edits on its own.

## Requirements

Python ≥ 3.12 and Node.js. The Python dependencies (FastAPI, PyTorch, NumPy,
ipykernel) are pinned in `pyproject.toml` / `uv.lock`.
