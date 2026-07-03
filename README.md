# Lamplighter

[![CI](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml/badge.svg)](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml)

A visual PyTorch workbench that lives inside your Jupyter kernel. Assemble an
`nn.Module` by wiring nodes on a canvas, hand the session your data, and train
from the browser — with live loss/accuracy curves — then pull the trained model
back into the notebook. Because the backend runs *in the kernel*, your data
never moves: the app holds references, and training executes **exactly the
generated code the preview panes show**. Nothing runs that you can't read.

## The workflow

```python
import lamplighter
import torch

sess = lamplighter.start()          # serve the app, open the editor

# 1. Build a model on the canvas (shapes are inferred live as you wire).
# 2. Register data — references, not copies:
sess.data(X=X, y=y)                 #    ...then pick X/y in the app's Data tab
# 3. Press ▶ Run in the Training tab — trains in this kernel, curves stream live.
# 4. The results are already here:
sess.model                          # the trained nn.Module
sess.history                        # per-epoch metrics, ready to plot
```

A full MNIST walkthrough is in [`example.ipynb`](example.ipynb).

## The three tabs

**Model** — drag nodes from the palette, wire pins, and watch shapes flow:
every badge shows the tensor each node *produces* (`N × 128` — `N` is the batch,
which models never fix), inferred by running the real layers on PyTorch's meta
device. Invalid wiring is flagged in place. The Inspector edits each node's
parameters and shows its parameter count with the arithmetic
(`100,480 parameters = 128×784 + 128`). Drop a node onto a wire to splice it in.
Inputs/Outputs can be named (named `forward` args, namedtuple returns);
multi-input and multi-output models are supported.

**Data** — a visual `DataLoader` constructor plus a pre-run checklist. Pick
registered tensors (shapes auto-fill the model's Input), or use torchvision
datasets (MNIST/CIFAR/…, with train-only augmentations) or an `ImageFolder`.
The **diagnostics pane** checks your actual data against the model before you
run: shape/dtype fit, X↔y alignment, loss↔target compatibility (including class
indices that would crash mid-run), batch-size traps like BatchNorm meeting a
ragged final batch.

**Training** — configure the loop (loss, optimizer, lr, epochs, device — only
devices your torch actually supports are offered), press **▶ Run**, and watch
loss/accuracy curves stream in per epoch. **■ Stop** ends a run early and keeps
the partial model. A tab opened mid-run picks the run up where it stands.
The loss chart rings the epoch with the **lowest validation loss** (`◦ best @k`);
those weights are captured as they happen and exposed as `sess.best_model`.
The **Checkpoints strip** keeps finished runs by name (in kernel memory):
**Restore** brings one back as the current run, **▶ Resume** trains further
from it — a warm start with the checkpoint's own graph and data picks, a fresh
optimizer, and a new recorded seed, epoch numbering continuing on one curve —
and ⬇ downloads it as a self-contained `.pt`. Set **Autosave Every** to roll a
resumable `autosave` checkpoint every N epochs, so stopping (or losing faith
in) a long run never costs the epochs already trained.

Every tab's **Show code** button reveals the generated source it drives — the
model, `make_dataloaders()`, and `train()` — and the Run button executes those
exact sources. **Export model.py** saves the model standalone.

## Nodes

| Category    | Nodes |
|-------------|-------|
| I/O         | Input, Output |
| Layers      | Linear, Embedding, Conv1d/2d/3d, MaxPool1d/2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, Dropout, Dropout2d, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RNN, LSTM, GRU |
| Activations | ReLU, Sigmoid, Tanh, LeakyReLU, GELU, ELU, SiLU, Softmax |
| Ops         | Concat |

Nodes are declarative registry data (`backend/registry.py`) — adding a layer is
one `NodeDef`; shape inference and code generation are generic over it.

## Notebook API

| Call | Description |
|------|-------------|
| `start(port=8000, ...)` | Start (or reuse) a session; returns a `Session`. |
| `sess.data(X=X, y=y)` | Register data references by name — merges across calls; re-register to repoint. |
| `sess.list_data()` / `sess.drop_data("X")` | Inspect / deregister. |
| `sess.model` / `sess.history` / `sess.run_status()` | Artifacts of the last app-triggered run. |
| `sess.best_model` | The model at the epoch with the lowest validation loss — often better than the (possibly overfit) final `sess.model`. |
| `sess.snapshot` | Full reproducibility record: seed, resolved device, configs, graph, and the exact sources that ran. |
| `sess.save_checkpoint("model.pt")` / `load_checkpoint(path)` | Save weights + snapshot as one self-contained file; reload the trained model anywhere — no session needed. `load_checkpoint(path, best=True)` picks the best-epoch weights. |
| `sess.checkpoint("name")` / `sess.checkpoints()` / `sess.restore("name")` | The in-app checkpoint store: keep the last run by name, list the entries, bring one back as the current run. |
| `sess.resume("name", epochs=None)` | Train further from a stored checkpoint (warm start; epoch numbering and the history continue). |
| `build_model()` | Instantiate the current canvas as an `nn.Module`. |
| `build_dataloaders()` | The Data tab's `make_dataloaders(X, y) -> (train_loader, val_loader)`. |
| `build_trainer()` | The Training tab's `train(model, loader, *, val_loader=None, on_epoch=None)` — returns a history dict; `on_epoch` gives per-epoch callbacks/early stopping. |
| `model_code()` / `data_code()` / `training_code()` | The generated sources, as strings. |
| `graph()` / `status()` / `open_editor()` / `stop()` | Session plumbing. |

Prefer owning the loop yourself? The generated pieces compose exactly like the
Run button does:

```python
model = lamplighter.build_model()
train_loader, val_loader = lamplighter.build_dataloaders()(X, y)
history = lamplighter.build_trainer()(model, train_loader, val_loader=val_loader)
```

## Architecture

Three parts, all local, one port:

- **Backend** (`backend/`) — FastAPI running on a daemon thread *inside the
  kernel*. Holds the graph, infers shapes on the meta device, generates all
  source, keeps the data registry (name → reference), runs pre-flight
  diagnostics, and executes training runs on a background thread with per-epoch
  progress pushed over the WebSocket.
- **Frontend** (`frontend/`) — React + [xyflow](https://reactflow.dev):
  palette, canvas, inspector, the Data and Training tabs, light/dark theme.
- **Client** (`lamplighter/`) — the notebook API and session lifecycle.

The graph lives in the backend, synced to every open tab over a WebSocket —
close a tab and reopen it, nothing is lost. Registry changes
(`sess.data(...)`) push to open tabs live.

## Development

```bash
uv sync                          # Python dependencies
uv run pytest                    # backend tests

cd frontend
npm install
npm run build                    # produces dist/, which the backend serves
npm test                         # frontend tests
```

The backend serves the built `dist/`, so after editing frontend source run
`npm run build` and hard-refresh. Backend edits need a kernel restart to take
effect in a running session (the kernel caches the imported modules).

## Requirements

Python ≥ 3.12 and Node.js. Python dependencies (FastAPI, PyTorch, torchvision,
NumPy, ipykernel) are pinned in `pyproject.toml` / `uv.lock`.
