# Lamplighter

[![CI](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml/badge.svg)](https://github.com/scott-kirila/Lamplighter/actions/workflows/ci.yml)

A visual PyTorch workbench that lives inside your Jupyter kernel. Assemble an
`nn.Module` by wiring nodes on a canvas, hand the session your data, and train
from the browser — with live loss/accuracy curves — then pull the trained model
back into the notebook. Because the backend runs *in the kernel*, your data
never moves: the app holds references, and training executes **exactly the
generated code the preview panes show**. Nothing runs that you can't read.

Build **one** model and train it supervised, or build **several** — connect them
in a high-level overview and train them together with a declarative recipe
(a GAN's generator + discriminator train in tandem, in-app).

## The workflow

```python
import lamplighter
import torch

sess = lamplighter.start()          # serve the app, open the editor

# 1. Build a model on the canvas (shapes are inferred live as you wire).
# 2. Register data — references, not copies:
sess.data(X=X, y=y)                 #    ...then pick X/y on the model's data node
# 3. Press ▶ Run in the Training tab — trains in this kernel, curves stream live.
# 4. The results are already here:
sess.model                          # the trained nn.Module
sess.history                        # per-epoch metrics, ready to plot
```

A full MNIST classifier walkthrough is in
[`examples/example.ipynb`](examples/example.ipynb); a two-model MNIST **GAN** is
in [`examples/gan.ipynb`](examples/gan.ipynb), a **conditional GAN** that
generates a digit you pick is in [`examples/cgan.ipynb`](examples/cgan.ipynb),
and a **VAE** whose latent space you can sample and interpolate is in
[`examples/vae.ipynb`](examples/vae.ipynb).

## The interface

**New project ▾** starts fresh — blank, or from a built-in template (MLP, CNN,
transformer classifier, GAN, VAE): a complete working project, pre-wired and
recipe-configured, held green by the test suite.

**Models** — the high-level dataflow canvas. Each model is a node you can
arrange, rename, and open (double-click, or the sidebar's **›**; the sidebar's
**＋** adds another). With several models, each opens in its own **subtab**
beside Models, so switching between them is one click — the **Models** tab
stays the wiring overview. **Data nodes** live here too: a **dataset** node
becomes a `DataLoader`, a **noise**
node an in-loop sampler — added from the sidebar's **DATA** palette and wired
into a model's input. Drag between any two nodes to **wire** them — a dataflow
claim that's shape-checked live (`Generator → Discriminator: N × 784`, or a red
edge when the source's output doesn't match the target's input; a data node's
fit is checked the same way). Select any node to configure it in the
**Inspector**. Data wiring is provisioned for you: a model's data-fed input gets
a dataset node (pick registered tensors — shapes auto-fill the model's Input — or
a torchvision dataset (MNIST/CIFAR/…, with train-only augmentations) or an
`ImageFolder`), and a GAN's generator gets a noise node whose latent size is the
source of truth for its Input. Both stay explicit — configurable, movable,
deletable.

**Inside a model** *(drill in from Models)* — drag nodes from the palette, wire
pins, and watch shapes flow: every badge shows the tensor each node *produces*
(`N × 128` — `N` is the batch, which models never fix), inferred by running the
real layers on PyTorch's meta device. Invalid wiring is flagged in place. The
Inspector edits each node's parameters and shows its parameter count with the
arithmetic (`100,480 parameters = 128×784 + 128`). Drop a node onto a wire to
splice it in. Inputs/Outputs can be named (named `forward` args, namedtuple
returns); multi-input and multi-output models are supported.

**Training** — pick a **recipe** (the training loop) and configure it. The
**Supervised** recipe is the classic loop (loss, optimizer, lr, epochs, device —
only devices your torch actually supports are offered). The **GAN (adversarial)**
recipe trains two models: assign the **Generator** and **Discriminator** roles
(each with its own learning rate), and it alternates discriminator/generator
steps under the hood — no target and no validation split. Recipes are
declarative data on the backend, so the loop is generated, not hand-picked. A
**readiness** checklist sits by ▶ Run, checking your actual registered data
against the model before you commit: shape/dtype fit, X↔y alignment, loss↔target
compatibility (including class indices that would crash mid-run), batch-size
traps like BatchNorm meeting a ragged final batch. Press **▶ Run** and the
metrics it reports stream into charts discovered from the run itself
(`train_loss`/`val_loss`, or a GAN's `g_loss`/`d_loss`). **■ Stop** ends a run
early and keeps the partial model(s); a tab opened mid-run picks the run up where
it stands.

The loss chart rings the epoch with the **lowest validation loss** (`◦ best @k`)
when the recipe has validation; those weights are captured as they happen and
exposed as `sess.best_model`.
The **Checkpoints strip** keeps runs by name (persisted to
`.lamplighter/checkpoints/` when autosave is on, so they survive a kernel
restart — weights load lazily, on first use):
**Restore** brings one back as the current run, **▶ Resume** continues one
toward its planned epoch target — an interrupted or autosaved run finishes its
plan in one click; a finished run takes a new, higher target. Resume is a warm
start: the checkpoint's own project and data picks, a fresh optimizer, a new
recorded seed, epoch numbering continuing on one curve. ⬇ downloads an entry
as a self-contained `.pt`. Set **Autosave Every** to roll a
resumable `autosave` checkpoint every N epochs, so stopping (or losing faith
in) a long run never costs the epochs already trained. Multi-model runs (a GAN)
checkpoint too — one `.pt` holds every model, and `load_checkpoint(path,
model="generator")` pulls one out by role.

Every tab's **Show code** button reveals the generated source it drives — the
model, `make_dataloaders()`, and `train()` — and the Run button executes those
exact sources. **Export model.py** saves the active model standalone.

## Nodes

| Category    | Nodes |
|-------------|-------|
| I/O         | Input, Output |
| Layers      | Linear, Embedding, Conv1d/2d/3d, MaxPool1d/2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, Dropout, Dropout2d, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RNN, LSTM, GRU, Self-Attention, Transformer Block, **Custom Module** (any `nn.Module` from your notebook, via `sess.modules(...)`) |
| Activations | ReLU, Sigmoid, Tanh, LeakyReLU, GELU, ELU, SiLU, Softmax |
| Ops         | Concat, Add (residual/skip connections), Reshape, Permute, Mean (sequence pooling) |

Nodes are declarative registry data (`backend/registry.py`) — adding a layer is
one `NodeDef`; shape inference and code generation are generic over it.

## Recipes

| Recipe | Roles | What it trains |
|--------|-------|----------------|
| Supervised | model | The classic loop: loss + optimizer over `(X, y)`, optional validation split and accuracy. |
| GAN (adversarial) | generator, discriminator | Alternating discriminator/generator steps (BCE on the real/fake decision); reports `g_loss`/`d_loss`. Latent noise is drawn to the generator's Input shape. |
| Conditional GAN | generator, discriminator | A GAN whose class label conditions both models — the dataset's `y` feeds each model's `label` port, so you can generate a *chosen* class. Reports `g_loss`/`d_loss`. |
| VAE (autoencoder) | encoder, decoder | Joint training with one optimizer: encode → reparameterize → decode, reconstruction (bce/mse) + `beta`·KL. The encoder exposes two *named* Outputs (`mu`, `logvar`). Reports `recon_loss`/`kl_loss`. |

Recipes are declarative too (`backend/recipes.py`) — a recipe is roles + form
params + a data contract + one `generate(project)` that emits the `train()`. The
runner and Training-tab form are generic over the registry, so adding a loop is
one `RecipeDef`, never a branch in an engine.

## Notebook API

| Call | Description |
|------|-------------|
| `start(port=8000, ...)` | Start (or reuse) a session; returns a `Session`. |
| `sess.data(X=X, y=y)` | Register data references by name — merges across calls; re-register to repoint. (A GAN registers just `X`.) |
| `sess.list_data()` / `sess.drop_data("X")` | Inspect / deregister. |
| `sess.modules(MyBlock=MyBlock)` | Register `nn.Module` *classes* for the **Custom Module** node — the palette escape hatch. The class source is spliced into generated code, so exports/checkpoints stay self-contained. |
| `sess.history` / `sess.run_status()` | Metrics + state of the last app-triggered run. |
| `sess.model` / `sess.models` | The trained model. `sess.models` is role → module (a GAN's `{"generator": …, "discriminator": …}`); `sess.model` is the sole module (None for a multi-model run — use `sess.models`). |
| `sess.best_model` | The model at the epoch with the lowest validation loss — often better than the (possibly overfit) final `sess.model`. None without validation (e.g. a GAN). |
| `sess.snapshot` | Full reproducibility record: seed, resolved device, configs, the project, and the exact sources that ran. |
| `sess.save_checkpoint("model.pt")` / `load_checkpoint(path)` | Save weights + snapshot as one self-contained file (every model, for a multi-model run); reload anywhere — no session needed. `load_checkpoint(path, best=True)` picks the best-epoch weights; `load_checkpoint(path, model="generator")` picks a model by role. |
| `sess.checkpoint("name")` / `sess.checkpoints()` / `sess.restore("name")` | The in-app checkpoint store: keep the last run by name, list the entries, bring one back as the current run. |
| `sess.resume("name", epochs=None)` | Continue a stored checkpoint toward its planned epoch target (finishes an interrupted run); `epochs` sets a new total to extend a finished one. Warm start; numbering and history continue. |
| `build_model()` | Instantiate the current canvas as an `nn.Module`. |
| `build_dataloaders()` | A dataset node's `make_dataloaders(X, y) -> (train_loader, val_loader)`. |
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
  kernel*. Holds the project (one or more models + how they connect), infers
  shapes on the meta device, generates all source, keeps the data registry
  (name → reference), runs pre-flight diagnostics, and executes training runs
  (single- or multi-model) on a background thread with per-epoch progress pushed
  over the WebSocket.
- **Frontend** (`frontend/`) — React + [xyflow](https://reactflow.dev):
  palette, per-model canvases, the Models overview, inspector, the Training tab,
  light/dark theme.
- **Client** (`lamplighter/`) — the notebook API and session lifecycle.

The project lives in the backend, synced to every open tab over a WebSocket —
close a tab and reopen it, nothing is lost. It's also autosaved to
`.lamplighter/graph.json` in the working directory on every edit and restored
at `start()`, so a kernel restart loses neither the project nor your named
checkpoints (`start(persist=False)` for scratch sessions). Registry changes
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

Notebooks under `examples/` are committed without outputs: run
`uv run nbstripout --install --attributes .gitattributes` once per clone to set
up the git filter — your working copies keep their outputs; git strips them at
staging time.

## Requirements

Python ≥ 3.12 and Node.js. Python dependencies (FastAPI, PyTorch, torchvision,
NumPy, ipykernel) are pinned in `pyproject.toml` / `uv.lock`.
