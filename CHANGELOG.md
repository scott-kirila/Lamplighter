# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org): the public surface is the notebook
API (`lamplighter.*`, `Session.*`) and the on-disk formats (`.lamplighter/`,
checkpoint `.pt`). The backend modules under `lamplighter/backend/` are the
documented extension surface but not a stable API before 1.0.

## 0.1.0

The first release. Lamplighter runs a FastAPI server on a daemon thread inside
your Jupyter kernel and reads your real registered tensors against your real
model, so it can tell you what will crash — a label off-by-one, a softmax under
CrossEntropyLoss, a ragged final batch meeting a BatchNorm — before you spend
the epoch. Build a model on the canvas or `sess.inspect` one you already have
(fx-traced, weight-seeded, runnable); train it in-kernel with live curves; pull
the trained model back into the notebook. Multi-model projects train together
under declarative recipes (GAN, cGAN, VAE); runs auto-record with a full
reproducibility snapshot and resumable checkpoints; hyperparameter sweeps
(`lamplighter[sweep]`) and reinforcement-learning recipes (`lamplighter[rl]`)
are optional extras. Ships as one wheel with the UI bundled — no Node toolchain
at install.

Everything below is the record of getting there; nothing prior shipped.

### Security

- The WebSocket handshake now checks `Origin` and every HTTP request checks
  `Host`, answering only to loopback. A browser can reach `127.0.0.1` on behalf
  of any page you visit, and WebSockets are exempt from the same-origin policy —
  so any page could previously open the editor socket, receive the whole project
  and its generated source, and replace it, a write that persisted to
  `.lamplighter/graph.json`. See `SECURITY.md`.

### Added

- `sess.inspect(model, x)` — trace an existing `nn.Module` onto the canvas,
  seeded with its original weights so you can run it, not just view it. Fidelity
  is checked, never faked: a layer the canvas can't represent exactly becomes an
  Opaque node, and a mostly-plumbing model (a transformer) is refused rather
  than drawn as holes; a clean import round-trips to numerically identical
  output (verified across the resnet family at maxdiff 0).
- `lamplighter.demo()` — one cell from install to an armed Run button, and an
  `mnist` template that trains with no notebook data at all (~15s to ~98% on a
  laptop CPU). Every other template needs tensors you supply, so a fresh install
  previously had no path from an empty canvas to a trained model.
- A Start panel on the empty canvas, offering that path instead of a dot grid.
- `lamplighter.diagnostics()` — versions, devices, installed extras, and whether
  the UI is served from the wheel or a checkout, in one paste-able block.
- `THIRD_PARTY_NOTICES.md`, generated from the installed licences and shipped in
  both the wheel and the sdist. The bundled JetBrains Mono (SIL OFL) and the
  runtime JS libraries have notice conditions the project's own MIT licence
  does not satisfy.
- Readiness now says what pressing Run will do about a torchvision download —
  already present, will fetch (with the size), or cannot work because Download
  is off and the files are absent.
- Readiness for an `ImageFolder` source now checks the folder exists and holds
  per-class subdirectories. The fine-tune template ships a placeholder root, so
  the panel went green on a path that need not exist and the run died inside
  `ImageFolder`'s constructor.

### Changed

- The readiness checklist survives the first run. It had one render site, gated
  on whether a run had streamed, so every check went dark for the life of the
  tab — during exactly the tweak-and-rerun loop where they matter. It is a
  collapsible strip now.
- `fastapi[standard]` → `fastapi` + `uvicorn[standard]`, dropping ~24 packages
  from every install including `sentry-sdk` and `fastapi-cloud-cli`.
- Node titles, filled controls and the bottom of the text ramp were rebuilt
  around measured contrast (node headers ran as low as 1.86:1; the faintest ramp
  step was 1.61:1). Ratios are now asserted against `index.css` in CI.
- Native controls follow the theme (`color-scheme`), so checkboxes and spinners
  stop being drawn from the OS light palette in a dark UI.

### Fixed

- A run whose loss diverged to NaN froze the dashboard while reporting success:
  the metric made the WebSocket frame unparseable and the REST fallback 500'd.
  Non-finite metrics now travel as `null`; `sess.history` keeps the raw value.
- A failed run discarded its traceback, keeping only a one-line summary — while
  generated sources are registered with `linecache` specifically so those frames
  resolve to real lines.
- A cached `index.html` survived an upgrade and pointed at a deleted bundle,
  giving a blank page with a hard refresh as the only cure.
- `Lamplighter(port=0)` returned a literal `0`, making every URL built from it
  unreachable.
- The run card's "in kernel" chip overflowed its column and painted over the
  save-weights button.
- The model canvas did not fit to view: the fit ran before nodes were measured,
  and React Flow's default `minZoom` clamped anything wider than ~10 nodes.
