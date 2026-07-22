"""Pre-run data↔model diagnostics for the Data tab's main pane.

Because the backend holds *references* to the registered data (see
``datastore``), it can check the real objects against the model before a run:
per-sample shapes vs the Input nodes, dtypes vs the Input dtype and the loss,
X↔y sample-count alignment, target/loss fit (including the class-range check
that otherwise surfaces as an opaque CUDA assert), and batch/val sanity.

Every check is a row: ``{"level": "ok" | "warn" | "error", "title", "detail"}``.
Pure over an injected namespace (defaults to the session registry) — the same
testability pattern as the rest of the data path.
"""
from __future__ import annotations

from typing import Any

from .inference import graph_issues
from .introspect import _arraylike_spec, input_shape_for, variable_kind
from .registry import default_data, default_training
from .schema import Graph, Project, resolve_data_config

# Canonical per-sample shapes of the curated torchvision datasets (C, H, W).
_CANONICAL: dict[str, list[int]] = {
    "MNIST": [1, 28, 28],
    "FashionMNIST": [1, 28, 28],
    "KMNIST": [1, 28, 28],
    "CIFAR10": [3, 32, 32],
    "CIFAR100": [3, 32, 32],
}

# Where each curated torchvision dataset lands under `root`, and roughly what it
# costs to fetch. The path is the dataset's own layout (what its `_check_exists`
# looks for), so an already-downloaded copy is recognised without constructing
# the dataset — diagnose runs on every edit and must not touch the network.
_TORCHVISION_FILES: dict[str, tuple[str, str]] = {
    "MNIST": ("MNIST/raw", "~11 MB"),
    "FashionMNIST": ("FashionMNIST/raw", "~30 MB"),
    "KMNIST": ("KMNIST/raw", "~20 MB"),
    "CIFAR10": ("cifar-10-batches-py", "~170 MB"),
    "CIFAR100": ("cifar-100-python", "~169 MB"),
}

_CLASSIFICATION_LOSSES = ("CrossEntropyLoss", "NLLLoss")

# Flag an imbalance once the biggest class outnumbers the smallest by this
# much — the point where an unweighted model starts winning by predicting the
# majority. A judgement call, deliberately loose: it's advice, not a blocker.
_IMBALANCE_RATIO = 3.0


def _row(level: str, title: str, detail: str = "") -> dict[str, str]:
    return {"level": level, "title": title, "detail": detail}


def source_mismatch(recipe: Any, source: str) -> tuple[str, str] | None:
    """(title, detail) when a recipe can't consume the wired source, else None.

    A loop that assumes a data SHAPE has to say so: fed the wrong one, a
    next-token recipe doesn't crash, it trains happily and reports a perplexity
    that describes nothing. Pure and shared, because both the pre-run checklist
    and the runner's start refusal must give the same verdict."""
    if getattr(recipe, "data", "loader") != "loader" or not recipe.data_sources:
        return None
    if source in recipe.data_sources:
        return None
    wanted = " or ".join(f"'{s}'" for s in recipe.data_sources)
    return (
        f"{recipe.label} can't train on the '{source}' source",
        f"it needs {wanted} — change Source on the dataset node, or pick another recipe",
    )


def _fmt(dims: list[int]) -> str:
    return "(" + ", ".join(str(d) for d in dims) + ")"


def _parse_input_shape(node) -> list[int] | None:
    try:
        dims = [int(t) for t in str(node.params.get("shape", "")).split(",") if t.strip()]
        return dims if dims else None
    except Exception:
        return None


def diagnose(project: Project, namespace: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Run the data↔model check suite for the project + registry — checks the
    recipe's data-fed model (a GAN's discriminator) and honors its contract (no
    target, no validation split for an adversarial loop)."""
    from .recipes import get_recipe

    if namespace is None:
        from .datastore import registry

        namespace = registry()

    recipe = get_recipe((project.training or {}).get("recipe"))
    needs_targets = recipe.needs_targets if recipe else True
    has_val = recipe.has_val if recipe else True
    data_role = recipe.data_role if recipe else "model"
    # A recipe without a user-facing loss knob bakes the loss into its loop (a
    # GAN's BCE on real/fake); its y is a conditioning input, not a supervised
    # target, so target↔loss fit doesn't apply.
    uses_loss = any(p.name == "loss" for p in recipe.params) if recipe else True

    if not project.models:
        return [_row("warn", "Empty canvas", "build a model on the Model tab first")]
    # The model that receives the real data X (the supervised model, or a GAN's
    # discriminator); fall back to the first model when roles aren't assigned.
    roles = (project.training or {}).get("roles") or {}
    mid = roles.get(data_role)
    model = next((m for m in project.models if m.id == mid), None) or project.models[0]
    data_config = resolve_data_config(project, model.id)
    graph = Graph(nodes=model.graph.nodes, edges=model.graph.edges)

    checks: list[dict[str, str]] = []
    data = {**default_data(), **data_config}
    training = {**default_training(), **(project.training or {})}
    source = str(data["source"])
    loss = str(training["loss"])

    if not graph.nodes:
        return [_row("warn", "Empty canvas", "build a model on the Model tab first")]

    # -- model side: report brokenness but keep running the data-only checks.
    from .codegen import generate_module, model_inputs
    from .inference import build_incoming, infer_shapes

    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    input_ids = model_inputs(graph, incoming, node_map)
    if not input_ids:
        # A broken model has no *live* inputs (nothing reaches a wired Output);
        # fall back to every Input node so the data-side checks still run.
        input_ids = sorted(
            (n.id for n in graph.nodes if n.type == "Input"),
            key=lambda nid: (node_map[nid].position.y, node_map[nid].position.x, nid),
        )

    model_output: list[int] | None = None
    shapes: dict = {}
    issues = graph_issues(graph)
    try:
        generate_module(graph)
        shapes, _ = infer_shapes(graph)
        outputs = [n.id for n in graph.nodes if n.type == "Output" and incoming.get(n.id)]
        if len(outputs) == 1:
            model_output = shapes.get((outputs[0], "output"))
        elif len(outputs) > 1:
            checks.append(_row("warn", "Multiple model outputs", "loss ↔ target fit isn't checked"))
    except ValueError as exc:
        checks.append(_row("error", "Model isn't ready", f"{exc} — fix it on the Model tab"))
    if issues:
        checks.append(_row("error", "Model has issues", "; ".join(issues)))

    # Recurrent layers set to seq-first contradict the batch-first pipeline —
    # only correct when you bring your own seq-first loader.
    from .codegen import _live_nodes

    for nid in _live_nodes(graph, incoming, node_map):
        n = node_map[nid]
        if n.type in ("RNN", "LSTM", "GRU") and n.params.get("batch_first") is False:
            checks.append(_row(
                "warn",
                f"{n.type} has batch_first=False but the pipeline feeds batch-first batches",
                "only correct with your own seq-first DataLoader — otherwise re-enable Batch First",
            ))

    # A pretrained backbone has expectations about its input that nothing else
    # in the pipeline announces.
    _check_backbones(checks, graph, incoming, node_map, shapes, data)

    # Next-token prediction has one catastrophic failure mode — attention that
    # can see the answer — and it looks like brilliant training.
    if recipe is not None and recipe.name == "causal_lm":
        _check_causal_lm(checks, graph, incoming, node_map, shapes, data, namespace, model_output)

    # The classic logits-vs-probabilities footgun (double-softmax / NLLLoss
    # without LogSoftmax) — only meaningful when the recipe exposes a loss knob.
    if uses_loss:
        _check_output_activation(checks, graph, loss, incoming, node_map)

    # An env recipe (RL): the environment is the data source — the dataset
    # checks below don't apply. Verify the wiring + the curated id + the
    # obs/action-space fit here.
    if recipe is not None and getattr(recipe, "data", "loader") == "env":
        from .recipes import RL_ENVS
        from .schema import resolve_env_config

        # The RL loop steps the policy ONE observation at a time, so layers
        # that need a batch break outright (BatchNorm wants n>1 in train mode)
        # and mask-resampling layers blur the policy (a fresh Dropout mask per
        # forward — GRPO's ratios then compare mismatched masks).
        for nid in _live_nodes(graph, incoming, node_map):
            n = node_map[nid]
            if n.type.startswith("BatchNorm"):
                checks.append(_row(
                    "error", f"{n.type} can't train on single observations",
                    "RL steps the policy one observation at a time — "
                    "remove it, or use LayerNorm (no batch statistics)",
                ))
            elif n.type.startswith("Dropout"):
                checks.append(_row(
                    "warn", f"{n.type} resamples its mask every forward",
                    "the action distribution wobbles, and GRPO's ratios compare "
                    "mismatched masks — remove it from the policy",
                ))

        _check_single_source(checks, project, model, "env")
        env_config = resolve_env_config(project, model.id)
        if env_config is None:
            checks.append(_row(
                "error", "No environment wired",
                "add one with ＋ env on the Models canvas and wire it into the policy",
            ))
        else:
            env_id = str(env_config.get("env_id", "") or "")
            if env_id not in RL_ENVS:
                checks.append(_row(
                    "error", f"unknown environment '{env_id}'",
                    f"expected one of: {', '.join(RL_ENVS)}",
                ))
            else:
                checks.append(_row("ok", f"environment: {env_id}"))
                _check_env_spaces(checks, env_id, input_ids, node_map, model_output)
        return checks

    # -- source-specific paths -------------------------------------------------
    _check_single_source(checks, project, model, "dataset")
    # A loop fed a shape it can't read is the one failure that looks like
    # success — refuse it here, before anything trains.
    mismatch = source_mismatch(recipe, source) if recipe is not None else None
    if mismatch is not None:
        checks.append(_row("error", *mismatch))
        return checks
    # The sampler is built by the in-memory loader path only — say so rather
    # than letting a stale toggle look active (the val_split lesson).
    if bool(data.get("weighted_sampler", False)):
        if source != "memory":
            checks.append(_row(
                "warn", f"the weighted sampler doesn't apply to the {source} source",
                "it balances an in-memory dataset — this toggle is ignored here",
            ))
        elif not needs_targets:
            checks.append(_row(
                "warn", "the weighted sampler needs labels to balance by",
                "this recipe trains on the inputs alone — the toggle is ignored",
            ))
    if source == "sequence":
        _check_sequence(checks, data, namespace, has_val)
        return checks
    if source == "torchvision":
        _check_torchvision(checks, data, input_ids, node_map)
        return checks
    if source == "imagefolder":
        _check_imagefolder(checks, data, input_ids, node_map)
        return checks

    # -- memory source: resolve each pick against the registry ------------------
    counts: dict[str, int] = {}
    tensor_inputs = 0
    loader_pick = False
    dataloader_pick = False

    for i, nid in enumerate(input_ids):
        node = node_map[nid]
        if len(input_ids) > 1:
            label = str(node.params.get("name") or "").strip() or f"Input {i}"
            name = str((data.get("x_vars") or {}).get(nid, "") or "").strip()
        else:
            label = "Input"
            name = str(data.get("x_var", "") or "").strip()

        if not name:
            checks.append(_row("error", f"{label}: nothing picked", "pick it on the dataset node (Models tab)"))
            continue
        if name not in namespace:
            checks.append(_row("error", f"{label}: '{name}' is not registered",
                               f"run sess.data({name}=...) in the notebook"))
            continue

        kind = variable_kind(name, namespace)
        if kind == "ndarray":
            checks.append(_row("error", f"{label}: '{name}' is a numpy array",
                               f"convert it with torch.from_numpy({name})"))
            continue
        if kind in ("dataset", "dataloader"):
            loader_pick = True
            dataloader_pick = dataloader_pick or kind == "dataloader"
            derived = input_shape_for(name, namespace)
            expected = _parse_input_shape(node)
            if derived and expected:
                sample = [int(t) for t in derived["shape"].split(",")][1:]
                if sample == expected[1:]:
                    checks.append(_row("ok", f"{label}: '{name}' sample {_fmt(sample)} matches"))
                else:
                    checks.append(_row("error", f"{label}: '{name}' sample {_fmt(sample)} ≠ Input {_fmt(expected[1:])}",
                                       "re-pick to auto-fill the Input shape"))
            note = "the loader owns batching and targets" if kind == "dataloader" \
                else "targets come from the dataset"
            checks.append(_row("ok", f"{label}: '{name}' is a {kind}", note))
            continue

        # tensor pick
        spec = _arraylike_spec(namespace[name])
        if spec is None:
            checks.append(_row("error", f"{label}: '{name}' has no usable shape"))
            continue
        dims, is_int = spec
        if not dims:
            checks.append(_row("error", f"{label}: '{name}' is a scalar tensor"))
            continue
        tensor_inputs += 1
        counts[name] = dims[0]
        sample = dims[1:]

        expected = _parse_input_shape(node)
        if expected is not None:
            if sample == expected[1:]:
                # Spell out the batch-dim reading: N samples of (per-sample dims).
                checks.append(_row(
                    "ok", f"{label}: '{name}' — {dims[0]} samples of {_fmt(sample)} match the Input"
                ))
            elif sample == expected:
                # The Input shape equals the data's per-sample shape — its leading
                # batch placeholder is missing (Input shapes start with one).
                checks.append(_row(
                    "warn",
                    f"{label}: the Input shape {_fmt(expected)} is missing its leading batch placeholder",
                    f"Input shapes start with a placeholder batch dim — set it to (1, "
                    f"{', '.join(map(str, sample))}), or re-pick '{name}' to auto-fill",
                ))
            elif dims == expected[1:]:
                # The data equals ONE sample's shape — it has no batch dim at all.
                checks.append(_row(
                    "error",
                    f"{label}: '{name}' {_fmt(dims)} looks like a single sample, not a batch",
                    f"expected (N, {', '.join(map(str, expected[1:]))}) — stack your samples",
                ))
            else:
                checks.append(_row("error", f"{label}: '{name}' sample {_fmt(sample)} ≠ Input {_fmt(expected[1:])}",
                                   "re-pick it to auto-fill the Input shape, or fix the Input node"))
        want_int = node.params.get("dtype") == "long"
        if is_int != want_int:
            have, want = ("integer", "float") if is_int else ("float", "integer")
            checks.append(_row("error", f"{label}: '{name}' is {have} but the Input expects {want}",
                               "set the Input's Dtype (or re-pick to auto-fill)"))

    # A picked DataLoader arrives already sampled — checked here rather than
    # with the other imbalance rows, which only run for tensor picks.
    if bool(data.get("weighted_sampler", False)) and dataloader_pick:
        checks.append(_row(
            "warn", "the picked DataLoader owns its own sampling",
            "the weighted sampler is ignored — rebalance inside your loader, or use Class Weights",
        ))

    # multi-input alignment
    if len(counts) > 1 and len(set(counts.values())) > 1:
        pairs = ", ".join(f"{k}: {v}" for k, v in counts.items())
        checks.append(_row("error", "Inputs have different sample counts", pairs))

    # -- target ------------------------------------------------------------------
    n = min(counts.values()) if counts else None
    # Refer to the data by its registered name(s), not a hardcoded "X".
    x_label = f"'{next(iter(counts))}'" if len(counts) == 1 else "the inputs"
    x_have = "has" if len(counts) == 1 else "have"
    if not needs_targets and tensor_inputs:
        # An adversarial recipe learns the data distribution — images only.
        checks.append(_row("ok", "No target needed", "this recipe trains on the inputs alone"))
    if needs_targets and tensor_inputs:
        y_name = str(data.get("y_var", "") or "").strip()
        if not y_name:
            checks.append(_row("error", "Target: nothing picked", "pick a target on the dataset node (Models tab)"))
        elif y_name not in namespace:
            checks.append(_row("error", f"Target: '{y_name}' is not registered",
                               f"run sess.data({y_name}=...) in the notebook"))
        else:
            spec = _arraylike_spec(namespace[y_name])
            if spec is None:
                checks.append(_row("error", f"Target: '{y_name}' isn't a tensor"))
            else:
                y_dims, y_int = spec
                if n is not None and y_dims and y_dims[0] != n:
                    checks.append(_row(
                        "error",
                        f"{x_label} {x_have} {n} samples but '{y_name}' has {y_dims[0]}",
                        "they must align row-for-row",
                    ))
                elif n is not None:
                    checks.append(_row("ok", f"{n} samples — {x_label} and '{y_name}' aligned"))
                if uses_loss:
                    _check_loss_fit(checks, loss, y_name, namespace[y_name], y_dims, y_int, model_output)
                    _check_imbalance(checks, data, training, loss, y_name, namespace[y_name])

    # -- batching / split sanity ---------------------------------------------------
    if n is not None and not loader_pick:
        _check_batching(checks, graph, data, n, node_map, incoming, has_val)
    return checks


# What ImageNet-pretrained backbones were trained on. Below this the pretrained
# features stop being worth much — the early layers see almost no detail.
_BACKBONE_NATIVE = 224
_BACKBONE_MIN = 64


def _check_backbones(
    checks: list, graph: Graph, incoming: dict, node_map: dict, shapes: dict, data: dict | None = None
) -> None:
    """A pretrained backbone's expectations about its input, which nothing else
    in the pipeline announces: 3 channels, ImageNet-ish resolution, ImageNet
    normalization, and a one-time weights download."""
    from .codegen import _live_nodes
    from .registry import REGISTRY, BackboneEmit, backbone_parts

    backbones = [
        nid for nid in _live_nodes(graph, incoming, node_map)
        if isinstance(getattr(REGISTRY.get(node_map[nid].type), "emit", None), BackboneEmit)
    ]
    if not backbones:
        return

    for nid in backbones:
        node = node_map[nid]
        try:
            spec, pretrained, freeze = backbone_parts(REGISTRY[node.type], node.params)
        except ValueError as exc:
            checks.append(_row("error", str(exc)))
            continue

        # The shape actually reaching it, else the declared Input shape when an
        # Input feeds it directly (so a broken graph still gets the advice).
        source = incoming.get(nid, {}).get("input")
        shape = shapes.get(source) if source else None
        if shape is None and source and node_map.get(source[0], node).type == "Input":
            declared = _parse_input_shape(node_map[source[0]])
            shape = declared if declared else None

        if shape and len(shape) == 4:
            channels, h, w = shape[1], shape[2], shape[3]
            if channels != 3:
                checks.append(_row(
                    "error", f"{spec.ctor} needs 3-channel images but gets {channels}",
                    "expand grayscale to 3 channels (repeat the channel), or use a from-scratch Conv2d stack",
                ))
            elif min(h, w) < _BACKBONE_MIN:
                checks.append(_row(
                    "warn", f"{spec.ctor} sees {h}×{w} images (it was trained on {_BACKBONE_NATIVE}×{_BACKBONE_NATIVE})",
                    f"below ~{_BACKBONE_MIN}px the pretrained features carry little — set Resize on the dataset node",
                ))
            else:
                checks.append(_row(
                    "ok", f"{spec.ctor}: {'frozen' if freeze else 'fine-tuning all weights'}",
                    f"outputs {spec.features} features",
                ))

        if pretrained:
            checks.append(_row(
                "ok", f"{spec.ctor} weights download on first use",
                "cached afterwards (~/.cache/torch) — the first run may pause",
            ))
            _check_pretrained_normalization(checks, data or {}, spec.ctor)


def _check_pretrained_normalization(checks: list, data: dict, arch: str) -> None:
    """Pretrained weights were fitted to ImageNet-standardized inputs. Feeding
    them anything else asks about a distribution they never saw — no error, just
    quietly worse numbers, which is exactly the kind of thing a checklist should
    catch. Only meaningful where a transform pipeline exists to do it."""
    source = str(data.get("source", "memory"))
    if source not in ("torchvision", "imagefolder"):
        # In-memory tensors are standardized in the notebook, where they're made.
        return
    mode = str(data.get("normalize", "none") or "none")
    if mode == "imagenet":
        checks.append(_row("ok", "inputs are ImageNet-normalized", f"the statistics {arch} was trained with"))
    elif mode == "dataset":
        checks.append(_row(
            "warn", f"inputs use this dataset's statistics, but {arch} was trained on ImageNet's",
            "set Normalize to 'imagenet' — a pretrained backbone expects the scaling it learned with",
        ))
    else:
        checks.append(_row(
            "warn", f"inputs aren't normalized, but {arch} was trained on ImageNet-standardized images",
            "set Normalize to 'imagenet' on the dataset node",
        ))


def _check_causal_lm(
    checks: list, graph: Graph, incoming: dict, node_map: dict, shapes: dict,
    data: dict, namespace: dict, model_output: list[int] | None,
) -> None:
    """The three things that quietly ruin a language model: attention that can
    read ahead, an output that isn't a distribution over the vocabulary, and a
    vocabulary the embedding can't hold."""
    from .codegen import _live_nodes
    from .registry import REGISTRY

    live = _live_nodes(graph, incoming, node_map)

    # 1. Bidirectional attention under a next-token objective means every
    # position can read the token it's being asked to predict. Training looks
    # superb and the model has learned nothing — the worst kind of bug.
    for nid in live:
        node = node_map[nid]
        emit = getattr(REGISTRY.get(node.type), "emit", None)
        param = getattr(emit, "causal_param", None)
        if param is None:
            continue
        if bool(node.params.get(param, False)):
            checks.append(_row("ok", f"{node.type} masks the future", "each position only sees what came before"))
        else:
            checks.append(_row(
                "error",
                f"{node.type} can see the whole sequence, including the next token",
                "next-token prediction with unmasked attention trains on the answer — "
                "turn on Causal (mask the future) on this node",
            ))

    # 2. The model must score every token in the vocabulary at every position.
    vocab = None
    embeddings = [nid for nid in live if node_map[nid].type == "Embedding"]
    if embeddings:
        try:
            vocab = int(node_map[embeddings[0]].params.get("num_embeddings"))
        except (TypeError, ValueError):
            vocab = None
    if model_output is not None:
        if len(model_output) != 3:
            checks.append(_row(
                "error", f"the model outputs {_fmt(model_output[1:])} per sample, not per-position logits",
                "a language model predicts at EVERY position — keep the sequence "
                "(don't pool it away) and end with a Linear sized to the vocabulary",
            ))
        elif vocab is not None and model_output[-1] != vocab:
            checks.append(_row(
                "error", f"the model outputs {model_output[-1]} logits but the vocabulary is {vocab}",
                f"set the last layer's out_features to {vocab}",
            ))
        elif vocab is not None:
            checks.append(_row("ok", f"logits at every position over all {vocab} tokens"))

    # 3. The Input declares the sequence length the model is built for; the
    # loader hands it windows of block_size. Disagreeing isn't cosmetic — a
    # PositionalEmbedding sized to the shorter one indexes past its table.
    block = int(data.get("block_size", 128) or 128)
    for nid in live:
        if node_map[nid].type != "Input":
            continue
        declared = _parse_input_shape(node_map[nid])
        if declared and len(declared) == 2 and declared[1] != block:
            checks.append(_row(
                "error", f"the Input expects {declared[1]} tokens but the loader yields windows of {block}",
                f"set the Input shape to (1, {block}), or change Block Size to {declared[1]}",
            ))

    # 4. The token ids have to fit the embedding table. With registered TEXT the
    # vocabulary is knowable exactly, so the fix names the number.
    name = str(data.get("corpus_var", "") or "").strip()
    tokens = namespace.get(name) if name else None
    if isinstance(tokens, str) and vocab is not None:
        from .codegen import char_vocab

        size = len(char_vocab(tokens))
        if size != vocab:
            checks.append(_row(
                "error", f"'{name}' has {size} distinct characters but the Embedding holds {vocab}",
                f"set num_embeddings to {size} (and the head's out_features to match)",
            ))
        else:
            checks.append(_row("ok", f"the model covers all {size} characters of '{name}'"))
        return
    if tokens is not None and vocab is not None:
        try:
            hi, lo = int(tokens.max()), int(tokens.min())
        except Exception:
            return
        if lo < 0 or hi >= vocab:
            checks.append(_row(
                "error", f"'{name}' holds token ids {lo}…{hi} but the Embedding has {vocab}",
                f"set num_embeddings to {hi + 1} (your vocabulary size)",
            ))
        else:
            checks.append(_row("ok", f"'{name}': {len(tokens.flatten())} tokens, ids {lo}…{hi}"))


def _class_counts(y: Any) -> list[int] | None:
    """Per-class sample counts for a class-like target — integer labels, or a
    float target that only holds 0/1 (what BCEWithLogits takes). None for
    anything else (a regression target has no classes to count)."""
    try:
        import torch

        t = y.flatten()
        if t.dtype.is_floating_point and set(t.unique().tolist()) - {0.0, 1.0}:
            return None
        t = t.long()
        if int(t.min()) < 0:
            return None
        return torch.bincount(t).tolist()
    except Exception:
        return None


def _check_imbalance(
    checks: list, data: dict, training: dict, loss: str, y_name: str, y: Any
) -> None:
    """Class balance, and the two remedies for it: weighting the LOSS (the
    training form) and resampling the DATA (the dataset node). Reports the real
    counts — the check is only useful if it says how skewed, and by how much."""
    from .codegen import _WEIGHTABLE_LOSSES

    weights_on = bool(training.get("class_weights", False)) and loss in _WEIGHTABLE_LOSSES
    sampler_on = bool(data.get("weighted_sampler", False))
    counts = _class_counts(y)

    if sampler_on and counts is None:
        checks.append(_row(
            "error", "the weighted sampler needs class labels",
            f"'{y_name}' isn't integer classes — there's nothing to balance by",
        ))
    if weights_on and sampler_on:
        checks.append(_row(
            "warn", "class weights AND a weighted sampler are both on",
            "rare classes would be drawn more often AND counted for more — pick one",
        ))
    if counts is None:
        return

    present = [c for c in counts if c > 0]
    if len(present) < 2:
        return
    ratio = max(present) / min(present)
    if ratio < _IMBALANCE_RATIO:
        return
    spread = ", ".join(f"{i}: {c}" for i, c in enumerate(counts) if c)
    if weights_on or sampler_on:
        remedy = "class weights" if weights_on else "a weighted sampler"
        checks.append(_row("ok", f"classes are imbalanced ({ratio:.0f}:1) — {remedy} rebalances them", spread))
    else:
        # Only name remedies that are actually ON SCREEN. The sampler always is
        # (this check only runs for the in-memory source), but Class Weights is
        # gated to the losses that take such an argument — advice pointing at an
        # absent control is worse than no advice.
        offered = ["a Weighted Sampler (the dataset node)"]
        if loss in _WEIGHTABLE_LOSSES:
            offered.insert(0, "Class Weights (Training)")
        checks.append(_row(
            "warn", f"classes are imbalanced ({ratio:.0f}:1)",
            f"{spread} — consider {' or '.join(offered)}",
        ))


def _check_single_source(checks: list, project: Project, model, kind: str) -> None:
    """One data source of a kind per model: the resolver is first-wire-wins, so a
    second wired dataset (or env) node would silently lose — flag it instead."""
    wired: list = []
    seen: set[str] = set()
    for link in project.links:
        if link.source_data is None or link.target_model != model.id:
            continue
        dn = next(
            (d for d in project.data_nodes if d.id == link.source_data and d.kind == kind), None
        )
        if dn is not None and dn.id not in seen:
            seen.add(dn.id)
            wired.append(dn)
    if len(wired) > 1:
        names = ", ".join(f"'{d.name}'" for d in wired)
        checks.append(_row(
            "error",
            f"{len(wired)} {kind} nodes wired into {model.name}",
            f"{names} — only '{wired[0].name}' would be used; remove the extra wire",
        ))


def _check_batching(
    checks: list, graph: Graph, data: dict, n: int, node_map: dict, incoming: dict, has_val: bool = True
) -> None:
    """Batch/split arithmetic the loader will actually perform — including the
    BatchNorm × batch-of-1 crash, which is fully predictable from n, batch_size,
    val_split, and drop_last."""
    from .codegen import _live_nodes

    batch = int(data.get("batch_size", 32) or 0)
    # An adversarial recipe has no held-out split, whatever the data config says.
    val = float(data.get("val_split", 0.0) or 0.0) if has_val else 0.0
    test = float(data.get("test_split", 0.0) or 0.0) if has_val else 0.0
    drop_last = bool(data.get("drop_last", False))

    if batch < 1:
        checks.append(_row("error", f"batch_size {batch} — must be at least 1"))
        return
    if not 0 <= val < 1:
        checks.append(_row("error", f"val_split {val} — must be in [0, 1)"))
        return
    if not 0 <= test < 1:
        checks.append(_row("error", f"test_split {test} — must be in [0, 1)"))
        return
    if val + test >= 1:
        checks.append(_row(
            "error", f"val_split {val} + test_split {test} leaves nothing to train on",
            "together they must take less than all of it",
        ))
        return

    n_test = int(n * test)
    n_val = int(n * val)
    n_train = n - n_val - n_test
    if val > 0:
        if n_val == 0:
            checks.append(_row("warn", f"val_split {val} of {n} samples holds out 0",
                               "no validation will run"))
        elif n_train == 0:
            checks.append(_row("error", f"val_split {val} holds out all {n} samples",
                               "nothing left to train on"))
            return
        else:
            checks.append(_row("ok", f"val split holds out {n_val} of {n} samples"))
    if test > 0:
        if n_test == 0:
            checks.append(_row("warn", f"test_split {test} of {n} samples holds out 0",
                               "there'd be nothing to evaluate on"))
        else:
            checks.append(_row("ok", f"test split holds out {n_test} of {n} samples",
                               "never trained or tuned on — what Evaluate scores"))

    if batch > n_train:
        if drop_last:
            # Every batch is ragged, so every batch is dropped — the train
            # loader yields nothing and the loop divides by zero.
            checks.append(_row(
                "error",
                f"Drop Last with batch_size {batch} > {n_train} training samples leaves no batches",
                "the whole epoch would be dropped — turn Drop Last off or lower batch_size",
            ))
            return
        checks.append(_row("warn", f"batch_size {batch} exceeds the {n_train} training samples",
                           "every epoch is a single batch"))

    # BatchNorm needs >1 sample per training batch — a batch of 1 crashes. The
    # final batch's size is deterministic (n_train % batch_size), so predict it.
    bn_types = sorted({
        node_map[nid].type for nid in _live_nodes(graph, incoming, node_map)
        if node_map[nid].type.startswith("BatchNorm")
    })
    ragged = n_train % batch
    if bn_types:
        bn = "/".join(bn_types)
        if batch == 1:
            checks.append(_row("error", f"batch_size 1 with {bn} in the model",
                               "BatchNorm needs more than 1 sample per training batch"))
        elif ragged == 1 and not drop_last:
            checks.append(_row(
                "error",
                f"the final batch has 1 sample and the model contains {bn}",
                f"{n_train} % {batch} = 1 — this crashes in training; enable Drop Last "
                "or change batch_size",
            ))

    if drop_last and ragged and ragged / n_train >= 0.25:
        checks.append(_row("warn", f"Drop Last discards {ragged} of {n_train} training samples every epoch",
                           "the ragged final batch is a big share of your data"))


def _check_loss_fit(
    checks: list, loss: str, y_name: str, y: Any, y_dims: list[int], y_int: bool,
    model_output: list[int] | None,
) -> None:
    """Does the target actually fit the chosen loss (and the model's output)?"""
    if loss == "Custom":
        # A registered loss class: its target contract is the user's business,
        # so no dtype/shape rule applies. Say what IS knowable — the metric
        # specs gate on built-in losses, so none can be reported.
        checks.append(_row(
            "ok", "custom loss — target fit isn't checked",
            "shape/dtype rules are yours; per-epoch metrics report loss only",
        ))
        return
    if loss in _CLASSIFICATION_LOSSES:
        if not y_int:
            checks.append(_row("error", f"{loss} needs integer class targets but '{y_name}' is float",
                               f"e.g. sess.data({y_name}={y_name}.long())"))
            return
        if len(y_dims) != 1:
            detail = ""
            if len(y_dims) == 2 and y_dims[1] == 1:  # the (N, 1) column-vector classic
                detail = f"squeeze the extra dim: sess.data({y_name}={y_name}.squeeze(1))"
            checks.append(_row("error", f"{loss} expects 1-D class targets but '{y_name}' is {_fmt(y_dims)}", detail))
            return
        if model_output is not None and len(model_output) == 2:
            n_classes = model_output[-1]
            try:  # a real read of the registered tensor — cheap at notebook scale
                y_max, y_min = int(y.max()), int(y.min())
            except Exception:
                return
            if y_min < 0 or y_max >= n_classes:
                checks.append(_row(
                    "error",
                    f"'{y_name}' has classes {y_min}…{y_max} but the model outputs {n_classes}",
                    "this would crash mid-run — adjust the last layer's out_features",
                ))
            elif n_classes > y_max + 1:
                # Runs fine, but the extra logits can never be a right answer —
                # usually a forgotten out_features default on the last layer.
                checks.append(_row(
                    "warn",
                    f"the model outputs {n_classes} classes but '{y_name}' only uses {y_min}…{y_max}",
                    f"did you mean out_features={y_max + 1} on the last layer?",
                ))
            else:
                checks.append(_row("ok", f"classes {y_min}…{y_max} match the model's {n_classes} outputs"))
    else:
        if y_int:
            checks.append(_row("error", f"{loss} needs float targets but '{y_name}' is integer",
                               f"e.g. sess.data({y_name}={y_name}.float())"))
        elif model_output is not None and y_dims[1:] != model_output[1:]:
            checks.append(_row("warn", f"'{y_name}' sample {_fmt(y_dims[1:])} vs model output {_fmt(model_output[1:])}",
                               f"{loss} may broadcast unexpectedly"))


def _check_env_spaces(
    checks: list, env_id: str, input_ids: list, node_map: dict, model_output: list[int] | None
) -> None:
    """The env↔policy fit — the class-range check's RL sibling: the env's
    observation shape must match the policy's Input, and its action count the
    policy's output logits. Degrades to a warn when Gymnasium can't be asked."""
    from .inference import env_spaces

    spaces = env_spaces(env_id)
    if spaces is None:
        checks.append(_row(
            "warn", "environment spaces unavailable",
            'install Gymnasium to verify obs/action fit — pip install "lamplighter[rl]"',
        ))
        return
    obs_dims, n_actions = spaces

    if len(input_ids) == 1:
        expected = _parse_input_shape(node_map[input_ids[0]])
        if expected is not None:
            if expected[1:] == obs_dims:
                checks.append(_row("ok", f"observations {_fmt(obs_dims)} match the Input"))
            else:
                checks.append(_row(
                    "error",
                    f"{env_id} observes {_fmt(obs_dims)} but the Input is {_fmt(expected[1:])}",
                    f"set the Input shape to (1, {', '.join(map(str, obs_dims))})",
                ))

    if model_output is not None and len(model_output) == 2:
        n_logits = model_output[-1]
        if n_logits == n_actions:
            checks.append(_row("ok", f"{n_actions} action logits match {env_id}"))
        else:
            checks.append(_row(
                "error",
                f"the policy outputs {n_logits} logits but {env_id} has {n_actions} actions",
                f"set the last layer's out_features to {n_actions}",
            ))


def _check_output_activation(
    checks: list, graph: Graph, loss: str, incoming: dict, node_map: dict
) -> None:
    """CrossEntropyLoss and NLLLoss are the two losses people mis-pair with a final
    softmax. CrossEntropyLoss folds ``log_softmax`` into the loss, so it wants raw
    logits — a final Softmax/LogSoftmax double-counts it. NLLLoss wants
    log-probabilities, so it needs a LogSoftmax specifically (a plain Softmax is
    the wrong base). Checks the node feeding the model's sole wired Output; a
    multi-output model has no single loss-bearing head, so it's skipped (like the
    target↔loss fit)."""
    if loss not in _CLASSIFICATION_LOSSES:
        return
    wired = [n for n in graph.nodes if n.type == "Output" and incoming.get(n.id)]
    if len(wired) != 1:
        return
    src_id, _ = next(iter(incoming[wired[0].id].values()))
    final = node_map[src_id].type

    if loss == "CrossEntropyLoss":
        if final in ("Softmax", "LogSoftmax"):
            checks.append(_row(
                "error",
                f"CrossEntropyLoss expects raw logits but the model ends in {final}",
                "CrossEntropyLoss applies log-softmax internally — remove the final "
                f"{final} (feed raw logits), or switch the loss to NLLLoss for a "
                "LogSoftmax head",
            ))
        return
    # NLLLoss
    if final == "LogSoftmax":
        checks.append(_row("ok", "LogSoftmax → NLLLoss: log-probabilities match"))
    elif final == "Softmax":
        checks.append(_row(
            "error",
            "NLLLoss expects log-probabilities but the model ends in Softmax",
            "NLLLoss takes log-probabilities — use a LogSoftmax head instead of "
            "Softmax (or switch the loss to CrossEntropyLoss on raw logits)",
        ))
    else:
        checks.append(_row(
            "warn",
            f"NLLLoss expects log-probabilities but the model ends in {final}",
            "add a LogSoftmax before the Output, or switch the loss to "
            "CrossEntropyLoss (which takes raw logits)",
        ))


def _check_sequence(checks: list, data: dict, namespace: dict, has_val: bool = True) -> None:
    """The token stream feeding a next-token loader: it has to exist, be
    integer ids, and be long enough that every slice yields whole windows."""
    name = str(data.get("corpus_var", "") or "").strip()
    if not name:
        checks.append(_row("error", "Corpus: nothing picked",
                           "pick registered text — or a tensor of token ids — on the dataset node (Models tab)"))
        return
    if name not in namespace:
        checks.append(_row("error", f"Corpus: '{name}' is not registered",
                           f"run sess.data({name}=...) in the notebook"))
        return

    # Raw text is tokenized by the loader, which keeps the vocabulary — so the
    # sizes are knowable here, and samples can be read back as text later.
    if isinstance(namespace[name], str):
        _check_text(checks, data, name, namespace[name], has_val)
        return

    spec = _arraylike_spec(namespace[name])
    if spec is None:
        checks.append(_row("error", f"Corpus: '{name}' isn't a tensor"))
        return
    dims, is_int = spec
    if not is_int:
        checks.append(_row("error", f"Corpus: '{name}' is float, but token ids are integers",
                           f"convert it with {name}.long()"))
        return

    checks.append(_row("ok", f"'{name}': {int(dims[0]) if dims else 0} pre-tokenized ids",
                       "no vocabulary here, so samples read back as ids — register text to see words"))
    _check_window_fits(checks, data, int(dims[0]) if dims else 0, has_val)


def _check_text(checks: list, data: dict, name: str, text: str, has_val: bool) -> None:
    """Registered text: report the vocabulary it implies (the number the model's
    Embedding and head both have to match) and check it's long enough to window."""
    from .codegen import char_vocab

    vocab = char_vocab(text)
    if len(vocab) < 2:
        checks.append(_row("error", f"'{name}' has {len(vocab)} distinct characters",
                           "there's nothing to predict — register more varied text"))
        return
    checks.append(_row(
        "ok", f"'{name}': {len(text)} characters, vocabulary of {len(vocab)}",
        "tokenized by character; the vocabulary travels with the run, so samples read back as text",
    ))
    _check_window_fits(checks, data, len(text), has_val)


def _check_window_fits(checks: list, data: dict, n: int, has_val: bool) -> None:
    """The window arithmetic shared by both sequence picks: enough tokens to
    fill a window, in every slice that has to produce batches."""
    block = int(data.get("block_size", 128) or 128)
    if block < 1:
        checks.append(_row("error", f"block_size {block} — must be at least 1"))
        return
    val = float(data.get("val_split", 0.0) or 0.0) if has_val else 0.0
    test = float(data.get("test_split", 0.0) or 0.0) if has_val else 0.0
    if val + test >= 1.0:
        checks.append(_row("error", f"val_split {val} + test_split {test} leaves nothing to train on"))
        return
    n_val, n_test = int(n * val), int(n * test)
    n_train = n - n_val - n_test
    if n_train <= block:
        checks.append(_row(
            "error", f"{n_train} training tokens can't fill a {block}-token window",
            "register more text, or lower Block Size",
        ))
        return
    checks.append(_row(
        "ok", f"{n_train - block} training windows of {block} tokens",
        "each position predicts the next token",
    ))
    for label, size in (("validation", n_val), ("test", n_test)):
        if size and size <= block:
            checks.append(_row(
                "warn", f"the {label} slice holds {size} tokens — shorter than one {block}-token window",
                f"no {label} batches would be produced; raise the split or lower Block Size",
            ))
        elif size:
            checks.append(_row("ok", f"{size - block} {label} windows, from text after the training split"))


def _check_torchvision_download(checks: list, data: dict) -> None:
    """Is the data already here, and if not, what will pressing Run actually do?

    This is the one step in the zero-setup path that reaches the network, and it
    is silent: the run appears to hang while ~170 MB of CIFAR arrives, or it dies
    deep inside torchvision on a machine behind a proxy. Neither reads as "I am
    downloading". Say it before the run instead — and refuse outright when
    downloading is off and the files aren't there, which is a certain failure.
    """
    from pathlib import Path

    dataset = str(data.get("dataset", ""))
    known = _TORCHVISION_FILES.get(dataset)
    if known is None:
        return
    subdir, size = known
    root = Path(str(data.get("root") or "./data")).expanduser()
    present = (root / subdir).exists()

    if present:
        checks.append(_row("ok", f"{dataset} is already downloaded", f"reading from {root}"))
    elif data.get("download"):
        checks.append(_row(
            "warn", f"{dataset} will be downloaded on the first run ({size})",
            f"saved to {root} — later runs reuse it",
        ))
    else:
        checks.append(_row(
            "error", f"{dataset} isn't in {root} and Download is off",
            "turn Download on, or point Data Root at a copy you already have",
        ))


def _check_torchvision(checks: list, data: dict, input_ids: list, node_map: dict) -> None:
    _check_torchvision_download(checks, data)
    dataset = str(data.get("dataset", ""))
    canonical = _CANONICAL.get(dataset)
    if canonical is None or len(input_ids) != 1:
        return
    sample = list(canonical)
    resize = data.get("resize")
    if resize:
        sample = [sample[0], int(resize), int(resize)]
    expected = _parse_input_shape(node_map[input_ids[0]])
    if expected is None:
        return
    if expected[1:] == sample:
        checks.append(_row("ok", f"{dataset} samples {_fmt(sample)} match the Input"))
    else:
        checks.append(_row("error", f"{dataset} yields {_fmt(sample)} per sample but the Input is {_fmt(expected[1:])}",
                           f"set the Input shape to (1, {', '.join(map(str, sample))})"))
    checks.append(_row("ok", "validation uses the dataset's test split"))


def _check_imagefolder_root(checks: list, data: dict) -> None:
    """Does the picked folder exist and look like ImageFolder expects?

    ``ImageFolder`` wants one subdirectory per class (``root/cat/*.jpg``), and
    fails three distinct ways this can catch first: no path, no such directory,
    and a directory with no class subdirectories in it.
    """
    from pathlib import Path

    root = str(data.get("root") or "").strip()
    if not root:
        checks.append(_row("error", "Images: no folder picked",
                           "set Data Root to a directory of labelled images"))
        return

    path = Path(root).expanduser()
    if not path.exists():
        checks.append(_row("error", f"no such folder: {path}",
                           "set Data Root to a directory that exists — paths are relative "
                           "to the kernel's working directory"))
        return
    if not path.is_dir():
        checks.append(_row("error", f"{path} is a file, not a folder",
                           "ImageFolder reads a directory of per-class subdirectories"))
        return

    try:
        classes = sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError as exc:
        checks.append(_row("error", f"can't read {path}", str(exc)))
        return

    if not classes:
        checks.append(_row(
            "error", f"{path} has no class subdirectories",
            "ImageFolder reads one folder per class — e.g. "
            f"{path.name}/cat/*.jpg, {path.name}/dog/*.jpg",
        ))
        return

    shown = ", ".join(classes[:4]) + (f", +{len(classes) - 4} more" if len(classes) > 4 else "")
    checks.append(_row("ok", f"{len(classes)} classes in {path.name} — {shown}"))


def _check_imagefolder(checks: list, data: dict, input_ids: list, node_map: dict) -> None:
    # The folder itself, first. A template ships "./data" as a placeholder, so
    # without this the panel goes green on a path that doesn't exist and Run
    # dies in ImageFolder's constructor: exactly the mid-run surprise this
    # module exists to convert into a pre-flight sentence. It does NOT
    # short-circuit — the checks below are config arithmetic that holds whether
    # or not the folder is there, and one pass should surface everything.
    _check_imagefolder_root(checks, data)
    # Nobody has measured this folder's statistics, so "dataset" means nothing
    # here — codegen refuses it, and saying so pre-run beats a start error.
    if str(data.get("normalize", "none")) == "dataset":
        checks.append(_row(
            "error", "an image folder has no known statistics to normalize with",
            "use 'imagenet' (what a pretrained backbone expects) or 'none'",
        ))
    # The tree's size isn't knowable pre-run, so the batching arithmetic can't be
    # checked — but the split RANGE can (codegen refuses it with the same rule).
    val = float(data.get("val_split", 0.0) or 0.0)
    test = float(data.get("test_split", 0.0) or 0.0)
    if not 0.0 <= val < 1.0:
        checks.append(_row("error", f"val_split {val} — must be in [0, 1)"))
    if not 0.0 <= test < 1.0:
        checks.append(_row("error", f"test_split {test} — must be in [0, 1)"))
    elif val + test >= 1.0:
        checks.append(_row(
            "error", f"val_split {val} + test_split {test} leaves nothing to train on",
            "together they must take less than all of it",
        ))
    resize = data.get("resize")
    if not resize:
        checks.append(_row("warn", "ImageFolder images vary in size", "set Resize (px) so batches stack"))
        return
    if len(input_ids) != 1:
        return
    sample = [3, int(resize), int(resize)]
    expected = _parse_input_shape(node_map[input_ids[0]])
    if expected is None:
        return
    if expected[1:] == sample:
        checks.append(_row("ok", f"resized samples {_fmt(sample)} match the Input", "assumes RGB images"))
    else:
        checks.append(_row("error", f"resize yields {_fmt(sample)} per sample but the Input is {_fmt(expected[1:])}",
                           f"set the Input shape to (1, {', '.join(map(str, sample))})"))
