# Changelog

## 0.1.0 — Unreleased

First release candidate.

- Visual PyTorch model builder living inside the Jupyter kernel: wire an
  `nn.Module` on a canvas with live shape inference, hand the session your data
  by reference, train from the browser, pull the trained model back into the
  notebook. The app runs exactly the generated code its panels show.
- Multi-model projects with declarative training recipes — supervised, GAN,
  conditional GAN, VAE — wired on a dataflow overview (dataset/noise nodes,
  shape-checked links, per-model run history).
- First-class runs: every run auto-records (curves, per-layer training health,
  per-step loss, a full reproducibility snapshot: seed, device, configs, exact
  sources); keep weights to make a run restorable/resumable; checkpoints
  persist across kernel restarts and download as self-contained `.pt` files
  loadable anywhere via `lamplighter.load_checkpoint()`.
- Pre-run readiness diagnostics against the real registered tensors (shape and
  dtype fit, class-range vs. loss, BatchNorm batching traps,
  logits-vs-probabilities pairing).
- Ships as a single wheel with the built UI bundled — no Node toolchain needed
  at install time.
