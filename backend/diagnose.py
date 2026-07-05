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
from .schema import Graph, Project, project_from_graph, resolve_data_config

# Canonical per-sample shapes of the curated torchvision datasets (C, H, W).
_CANONICAL: dict[str, list[int]] = {
    "MNIST": [1, 28, 28],
    "FashionMNIST": [1, 28, 28],
    "KMNIST": [1, 28, 28],
    "CIFAR10": [3, 32, 32],
    "CIFAR100": [3, 32, 32],
}

_CLASSIFICATION_LOSSES = ("CrossEntropyLoss", "NLLLoss")


def _row(level: str, title: str, detail: str = "") -> dict[str, str]:
    return {"level": level, "title": title, "detail": detail}


def _fmt(dims: list[int]) -> str:
    return "(" + ", ".join(str(d) for d in dims) + ")"


def _parse_input_shape(node) -> list[int] | None:
    try:
        dims = [int(t) for t in str(node.params.get("shape", "")).split(",") if t.strip()]
        return dims if dims else None
    except Exception:
        return None


def diagnose(design: Graph | Project, namespace: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Run the data↔model check suite for the current design + registry. Accepts
    a single graph (one model) or a whole project (checks the recipe's data-fed
    model — a GAN's discriminator — and honors its contract: no target, no
    validation split)."""
    from .recipes import get_recipe

    if namespace is None:
        from .datastore import registry

        namespace = registry()

    project = design if isinstance(design, Project) else project_from_graph(design)
    recipe = get_recipe((project.training or {}).get("recipe"))
    needs_targets = recipe.needs_targets if recipe else True
    has_val = recipe.has_val if recipe else True
    data_role = recipe.data_role if recipe else "model"

    if not project.models:
        return [_row("warn", "Empty canvas", "build a model on the Model tab first")]
    # The model that receives the real data X (the supervised model, or a GAN's
    # discriminator); fall back to the first model when roles aren't assigned.
    roles = (project.training or {}).get("roles") or {}
    mid = roles.get(data_role)
    model = next((m for m in project.models if m.id == mid), None) or project.models[0]
    data_config = resolve_data_config(project, model.id)
    graph = Graph(
        nodes=model.graph.nodes,
        edges=model.graph.edges,
        training=project.training,
        data=data_config,
    )

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

    # -- source-specific paths -------------------------------------------------
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

    for i, nid in enumerate(input_ids):
        node = node_map[nid]
        if len(input_ids) > 1:
            label = str(node.params.get("name") or "").strip() or f"Input {i}"
            name = str((data.get("x_vars") or {}).get(nid, "") or "").strip()
        else:
            label = "Input"
            name = str(data.get("x_var", "") or "").strip()

        if not name:
            checks.append(_row("error", f"{label}: nothing picked", "pick registered data on the left"))
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
            checks.append(_row("error", "Target: nothing picked", "pick a target on the left"))
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
                _check_loss_fit(checks, loss, y_name, namespace[y_name], y_dims, y_int, model_output)

    # -- batching / split sanity ---------------------------------------------------
    if n is not None and not loader_pick:
        _check_batching(checks, graph, data, n, node_map, incoming, has_val)
    return checks


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
    drop_last = bool(data.get("drop_last", False))

    if batch < 1:
        checks.append(_row("error", f"batch_size {batch} — must be at least 1"))
        return
    if not 0 <= val < 1:
        checks.append(_row("error", f"val_split {val} — must be in [0, 1)"))
        return

    n_val = int(n * val)
    n_train = n - n_val
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

    if batch > n_train:
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


def _check_torchvision(checks: list, data: dict, input_ids: list, node_map: dict) -> None:
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


def _check_imagefolder(checks: list, data: dict, input_ids: list, node_map: dict) -> None:
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
