# Recording the launch assets

The one thing that has to be recorded by hand — a screen capture can't be
scripted here. This is the shot list. The pitch is **verification**, so every
asset opens on a problem and ends on the catch, never on "look, a canvas".

## The hero GIF (the one that goes at the top of the README)

**~20 seconds, silent, seamless loop, ~1000px wide, ≤ 3 MB, burned-in captions.**
Silent autoplay is the only playback GitHub gives a README GIF, so the captions
carry it.

Lead with **import + verify**, not build. A 2026 viewer has already priced
"assemble a model by wiring nodes" — their reflex is "I'll ask an agent". A
canvas-building GIF confirms that reflex; catching a bug in code that already
exists challenges it. So:

| t (s) | On screen | Caption (bottom third) |
|-------|-----------|------------------------|
| 0–3 | A notebook cell: `sess.inspect(models.resnet18(), input_shape=(1,3,224,224))` runs; the model appears on the canvas. | "Bring a model you already have." |
| 3–6 | Cut to the Training tab. `sess.data(X=X, y=y)` has registered EMNIST-letters-shaped data; the Pre-flight panel is visible. | "Point it at your data." |
| 6–11 | The Pre-flight panel, full size: two green ✓ rows, then the red **✗ `y` has classes 1…26 but the model outputs 26**. ▶ Run is disabled. Hold here — this is the frame people screenshot. | "It reads your real tensors. `1…26` is arithmetic, not a guess." |
| 11–15 | One edit: the last Linear's *Out Features* 26 → 27 (or `y = y - 1` in the notebook). The red row flips green. ▶ Run enables. | "Fix it, and it un-blocks." |
| 15–20 | ▶ Run. Loss/accuracy curves stream. | "Then run — in your kernel, exactly the code it shows." |

The money frame is 6–11s: two independently-checkable facts on one screen, the
Run button dark. That single still is `docs/assets/preflight-catch.jpg` (already
captured) — re-shoot it at retina resolution for the final GIF.

### Reproducing the bug (real, third-party, deterministic)

```python
from torchvision import datasets, transforms
emnist = datasets.EMNIST(root='./data', split='letters', train=True,
                         download=True, transform=transforms.ToTensor())
```

EMNIST `letters` labels are **1…26** (torchvision does not remap them). Wire an
MLP/any head to `out_features=26`, pick `CrossEntropyLoss`, and
`_check_loss_fit` fires exactly. No planted bug — a viewer can reproduce it.
`examples/verify.ipynb` is this setup end to end.

## The five README stills

Retina, cropped tight, dark theme.

1. **The Pre-flight catch** — the panel with the red row, ▶ Run disabled.
   *Caption: "Two crashes, found before the run."* → `docs/assets/preflight-catch.jpg` ✓ captured.
2. **An imported resnet18 on the canvas** — the 71-node graph, laid out.
   *Caption: "Bring a model you already have — and run it."*
3. **The Training dashboard mid-run** — loss/accuracy curves + the epoch table.
   *Caption: "Then train, in your kernel."*
4. **The training-health badge** — a layer node reading `Δw/w = … · N× below the fastest layer` on a saturating-sigmoid stack. *Caption: "It keeps watching after the run starts."*
5. **Show code** — the generated `train()` beside the canvas.
   *Caption: "Nothing runs that you can't read."*

### Still 4's setup (reproducible vanishing gradient)

`Input(784) → Linear(128) → Sigmoid → Linear(128) → Sigmoid → Linear(128) →
Sigmoid → Linear(10)`, SGD lr 0.01, an MNIST subsample, ~8 epochs. The sigmoids
saturate; the first Linear's ‖Δw‖/‖w‖ lands orders below the last layer's and
the health badge goes full red with a real number. Record a real run — don't
stage the number.

## Tools

- **macOS**: `⌘⇧5` for the region, or [Kap](https://getkap.co) straight to GIF.
- Hide the cursor except where it matters; slow the pointer.
- 1000px wide, then `gifsicle -O3 --lossy=80` (or an ffmpeg palette pass) to get
  under GitHub's comfortable inline size.
- Prefer an MP4 uploaded to the repo's releases for the long (60-90s) tour and
  link it; keep the GIF to the 20s hero loop.

## The Colab notebook

A kernel-local tool can't be hosted, but Colab can run the kernel *and* surface
its UI. `serve_kernel_port_as_window(port)` renders the in-kernel server at a
clickable, authenticated URL — which satisfies Show HN's "something people can
play with without signups". `examples/colab.ipynb` is the hosted playground; keep
it to `demo()` plus the import-and-verify beat so it opens in one run.
