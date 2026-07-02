from .schema import Graph
from .inference import infer_shapes, build_incoming, topo_order
from .registry import REGISTRY, ModuleEmit, default_data, default_training, render_module_args


def _live_nodes(graph: Graph, incoming: dict, node_map: dict) -> set[str]:
    """The subgraph that feeds the wired Output(s) — nodes backward-reachable
    from them. Stray/disconnected nodes are excluded so a scratch node on the
    canvas doesn't affect codegen."""
    outputs = [n.id for n in graph.nodes if n.type == "Output" and incoming.get(n.id)]
    live: set[str] = set()
    stack = list(outputs)
    while stack:
        nid = stack.pop()
        if nid in live:
            continue
        live.add(nid)
        stack.extend(src for src, _handle in incoming.get(nid, {}).values())
    return live


def model_inputs(graph: Graph, incoming: dict, node_map: dict) -> list[str]:
    """Live Input node ids ordered by canvas position (top-to-bottom, then
    left-to-right), so each maps to a forward() argument in visual order. Shared
    by module and training codegen so the arg count/order can't diverge."""
    live = _live_nodes(graph, incoming, node_map)
    ins = [n.id for n in graph.nodes if n.type == "Input" and n.id in live]
    return sorted(ins, key=lambda nid: (node_map[nid].position.y, node_map[nid].position.x, nid))


def _node_name(node) -> str:
    """The user-set `name` param of an Input/Output node, stripped (blank = auto)."""
    return str(node.params.get("name", "") or "").strip()


def _input_arg_names(inputs: list[str], node_map: dict) -> list[str]:
    """forward() argument names for the ordered Input nodes. A node's `name` param
    is used verbatim when set; otherwise a lone input stays `x` (byte-identical to
    the unnamed single-input case) and several become `x0, x1, …`."""
    return [
        _node_name(node_map[nid]) or ("x" if len(inputs) == 1 else f"x{i}")
        for i, nid in enumerate(inputs)
    ]


def _output_field_names(outputs: list[str], node_map: dict) -> list[str]:
    """namedtuple field names for the ordered Output nodes — the `name` param when
    set, else out0/out1/… so a partially-named model still resolves."""
    return [_node_name(node_map[nid]) or f"out{i}" for i, nid in enumerate(outputs)]


def _history_keys(track_acc: bool, include_val: bool) -> list[str]:
    """Ordered metric keys for the returned per-epoch history dict."""
    keys = ["train_loss"]
    if track_acc:
        keys.append("train_acc")
    if include_val:
        keys.append("val_loss")
        if track_acc:
            keys.append("val_acc")
    return keys


def _history_init_line(keys: list[str]) -> str:
    """`    history = {"train_loss": [], …}` — one empty list per tracked metric."""
    return "    history = {" + ", ".join(f'"{k}": []' for k in keys) + "}"


def _device_resolution_lines() -> list[str]:
    """Generated preamble that turns the `device` arg into a torch.device and moves
    the model onto it. "auto" prefers CUDA, then MPS (guarded for torch builds
    without the mps backend), else CPU; a specific name is used as-is."""
    return [
        '    if device == "auto":',
        "        if torch.cuda.is_available():",
        '            device = "cuda"',
        '        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():',
        '            device = "mps"',
        "        else:",
        '            device = "cpu"',
        "    device = torch.device(device)",
        "    model = model.to(device)",
    ]


def generate_module(graph: Graph) -> str:
    shapes, errors = infer_shapes(graph)

    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    order, _ = topo_order(graph, incoming)

    # Wired Output nodes — each contributes a value to the model's return. An
    # unwired Output is ignored (a scratch node), like any other stray node.
    outputs = [nid for nid in order if node_map[nid].type == "Output" and incoming.get(nid)]
    if not outputs:
        raise ValueError("expected at least 1 connected Output node, found 0")

    # The model is the subgraph that feeds the Output(s) — the nodes backward-
    # reachable from them. Stray/disconnected nodes (and any errors they carry)
    # are ignored, so a scratch node on the canvas doesn't break codegen.
    live = _live_nodes(graph, incoming, node_map)

    live_errors = {k: v for k, v in errors.items() if k in live}
    if live_errors:
        detail = "; ".join(f"{k}: {v}" for k, v in live_errors.items())
        raise ValueError(f"Graph has errors — {detail}")

    inputs = model_inputs(graph, incoming, node_map)
    if not inputs:
        raise ValueError("expected at least 1 Input node, found 0")
    arg_names = _input_arg_names(inputs, node_map)

    # Each (node, output pin) gets an SSA variable; each Input maps to a forward
    # arg (x, or x0/x1/… for several). `used_pins` is the set of pins wired
    # downstream — unwired outputs of a multi-output node are never materialized.
    used_pins = {(e.source, e.sourceHandle) for e in graph.edges}
    var: dict[tuple[str, str], str] = {
        (nid, "output"): name for nid, name in zip(inputs, arg_names)
    }
    output_vars: dict[str, str] = {}  # wired Output node id -> the var it returns
    counter = 0
    midx = 0

    init_lines: list[str] = []
    fwd_lines: list[str] = []

    def sv(nid: str, handle: str = "input") -> str:
        return var[incoming[nid][handle]]

    for nid in order:
        if nid not in live:
            continue  # stray node — not part of the model
        node = node_map[nid]
        t = node.type
        p = node.params

        if t == "Input":
            continue
        if t == "Output":
            output_vars[nid] = var[next(iter(incoming[nid].values()))]
            continue

        if t == "Concat":
            handles = sorted(incoming[nid])
            args = ", ".join(var[incoming[nid][h]] for h in handles)
            dim = int(p.get("dim", 1))
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = torch.cat([{args}], dim={dim})")
            continue

        # Standard nodes render an nn.<cls> member + call, built from the same
        # args inference uses (so code and shapes can't disagree).
        node_def = REGISTRY.get(t)
        emit = node_def.emit if node_def else None

        if isinstance(emit, ModuleEmit):
            input_shape = shapes[incoming[nid]["input"]]
            rendered = render_module_args(node_def, p, input_shape)
            init_lines.append(f"self.layer_{midx} = nn.{emit.cls}({rendered})")
            result = f"t{counter}"
            counter += 1
            fwd_lines.append(f"{result} = self.layer_{midx}({sv(nid)})")
            midx += 1
            # Materialize each wired output pin from the return value: a single
            # tensor return (path ()) is the result itself; multi-output layers
            # index into it (e.g. LSTM's `output` = result[0]).
            for pin, path in emit.outputs:
                if (nid, pin) not in used_pins:
                    continue
                if path == ():
                    var[(nid, pin)] = result
                else:
                    access = result + "".join(f"[{i}]" for i in path)
                    extracted = f"t{counter}"
                    counter += 1
                    fwd_lines.append(f"{extracted} = {access}")
                    var[(nid, pin)] = extracted

    # Return each wired Output's value, ordered top-to-bottom by canvas position
    # (so a multi-output model's tuple/field order matches the visual layout). A
    # multi-output model with any named Output returns a namedtuple so callers can
    # unpack it *and* access fields by name; otherwise it's a plain tuple.
    ordered = sorted(outputs, key=lambda nid: (node_map[nid].position.y, node_map[nid].position.x, nid))
    named = len(ordered) > 1 and any(_node_name(node_map[nid]) for nid in ordered)
    fields = _output_field_names(ordered, node_map) if named else []

    header = ["import torch", "import torch.nn as nn"]
    if named:
        field_list = ", ".join(repr(f) for f in fields)
        header += [
            "from collections import namedtuple",
            "",
            "",
            f'ModelOutput = namedtuple("ModelOutput", [{field_list}])',
        ]

    parts = header + [
        "",
        "",
        "class GeneratedModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    parts += ["        " + line for line in (init_lines or ["pass"])]
    parts += ["", f"    def forward(self, {', '.join(arg_names)}):"]
    parts += ["        " + line for line in (fwd_lines or ["pass"])]
    if named:
        args = ", ".join(f"{f}={output_vars[nid]}" for f, nid in zip(fields, ordered))
        parts.append(f"        return ModelOutput({args})")
    else:
        parts.append(f"        return {', '.join(output_vars[nid] for nid in ordered)}")

    return "\n".join(parts) + "\n"


def generate_training(graph: Graph) -> str:
    """A self-contained ``train(model, loader)`` from the graph's training config
    (loss/optimizer/hyperparams, metric, device). Data always arrives as a torch
    DataLoader built by the Data panel's ``make_dataloaders()`` — one data path,
    so what runs is exactly what both panels show. An optional ``val_loader``
    runs validation; ``on_epoch`` reports per-epoch metrics and supports early
    stopping (return False to stop)."""
    cfg = {**default_training(), **(graph.training or {})}
    loss = str(cfg["loss"])
    optimizer = str(cfg["optimizer"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    metric = str(cfg["metric"])
    device = str(cfg["device"])

    # Top-1 (argmax) accuracy is only meaningful for classification losses, so
    # gate it on the loss — a regression loss never emits accuracy code.
    track_acc = metric == "accuracy" and loss in ("CrossEntropyLoss", "NLLLoss")

    # A multi-input model's loader yields (x0, x1, …, y): `*xb, yb = batch`
    # unpacks the trailing target, the rest feed model(*xb).
    incoming = build_incoming(graph)
    node_map = {n.id: n for n in graph.nodes}
    multi = len(model_inputs(graph, incoming, node_map)) > 1

    opt_args = [f"lr={lr!r}"]
    if weight_decay != 0.0:  # omit the default for cleaner code
        opt_args.append(f"weight_decay={weight_decay!r}")
    opt_call = f"torch.optim.{optimizer}(model.parameters(), {', '.join(opt_args)})"

    if multi:
        unpack, to_dev, call = "*xb, yb = batch", "xb = [t.to(device) for t in xb]", "model(*xb)"
    else:
        unpack, to_dev, call = "xb, yb = batch", "xb = xb.to(device)", "model(xb)"

    lines = ["import torch", "import torch.nn as nn", "", ""]
    lines.append(
        f"def train(model, loader, *, epochs={epochs}, val_loader=None, device={device!r}, on_epoch=None):"
    )
    lines += _device_resolution_lines()
    # Val keys are always present (val_loader may be passed at call time); their
    # lists stay empty when no val_loader is given.
    lines += [
        f"    loss_fn = nn.{loss}()",
        f"    opt = {opt_call}",
        _history_init_line(_history_keys(track_acc, include_val=True)),
        "    for epoch in range(epochs):",
        "        model.train()",
        "        running, seen = 0.0, 0",
    ]
    if track_acc:
        lines.append("        correct = 0")
    lines += [
        "        for batch in loader:",
        f"            {unpack}",
        f"            {to_dev}",
        "            yb = yb.to(device)",
        "            opt.zero_grad()",
        f"            out = {call}",
        "            loss = loss_fn(out, yb)",
        "            loss.backward()",
        "            opt.step()",
        "            bs = yb.size(0)",
        "            running += loss.item() * bs",
        "            seen += bs",
    ]
    if track_acc:
        lines.append("            correct += (out.argmax(dim=-1) == yb).sum().item()")
    lines.append("        train_loss = running / seen")
    if track_acc:
        lines.append("        train_acc = correct / seen")
    # The report is built at run time because val is optional (val_loader=None).
    lines.append('        msg = f"epoch {epoch + 1}/{epochs}  loss {train_loss:.4f}"')
    if track_acc:
        lines.append('        msg += f" acc {train_acc:.3f}"')

    lines += [
        "        if val_loader is not None:",
        "            model.eval()",
        "            vloss, vseen = 0.0, 0",
    ]
    if track_acc:
        lines.append("            vcorrect = 0")
    lines += [
        "            with torch.no_grad():",
        "                for batch in val_loader:",
        f"                    {unpack}",
        f"                    {to_dev}",
        "                    yb = yb.to(device)",
        f"                    out = {call}",
        "                    bs = yb.size(0)",
        "                    vloss += loss_fn(out, yb).item() * bs",
        "                    vseen += bs",
    ]
    if track_acc:
        lines.append("                    vcorrect += (out.argmax(dim=-1) == yb).sum().item()")
    lines.append("            val_loss = vloss / vseen")
    if track_acc:
        lines.append("            val_acc = vcorrect / vseen")
    lines.append('            msg += f"  val_loss {val_loss:.4f}"')
    if track_acc:
        lines.append('            msg += f" val_acc {val_acc:.3f}"')
    # Val metrics recorded only on epochs where a val_loader ran.
    lines.append('            history["val_loss"].append(val_loss)')
    if track_acc:
        lines.append('            history["val_acc"].append(val_acc)')

    lines.append("        print(msg)")
    lines.append('        history["train_loss"].append(train_loss)')
    if track_acc:
        lines.append('        history["train_acc"].append(train_acc)')
    # Per-epoch hook: progress reporting and early stopping (return False to stop).
    lines.append("        if on_epoch is not None and on_epoch(epoch + 1, history) is False:")
    lines.append("            break")
    lines.append("    return history")
    return "\n".join(lines) + "\n"


def generate_dataloader(graph: Graph, namespace: dict | None = None) -> str:
    """A `make_dataloaders()` helper from the Data panel's config, returning
    (train_loader, val_loader). It pairs with the DataLoader training mode:
    `train_loader, val_loader = make_dataloaders(...)` then
    `train(model, train_loader, val_loader=val_loader)`. `namespace` (the live
    kernel vars, injectable for tests) lets the `variable` source specialize by
    the picked object's type."""
    cfg = {**default_data(), **(graph.data or {})}
    source = str(cfg["source"])
    batch_size = int(cfg["batch_size"])
    shuffle = bool(cfg["shuffle"])
    # Only the train loader drops a ragged batch; omitted when off for clean code.
    drop = ", drop_last=True" if bool(cfg["drop_last"]) else ""
    common = _loader_common(cfg)  # num_workers / pin_memory, on every loader
    # A multi-input model needs make_dataloaders(X0, X1, y) → TensorDataset(X0, X1, y).
    n_inputs = sum(1 for n in graph.nodes if n.type == "Input") or 1
    if source == "torchvision":
        return _dataloader_torchvision(cfg, batch_size, shuffle, drop, common)
    if source == "imagefolder":
        return _dataloader_imagefolder(cfg, batch_size, shuffle, drop, common)
    return _dataloader_memory(cfg, batch_size, shuffle, drop, common, namespace, n_inputs)


def _loader_common(cfg: dict) -> str:
    """Non-default DataLoader perf kwargs (num_workers/pin_memory) as a `, k=v…`
    suffix applied to every loader; empty when both are at defaults."""
    parts = []
    if int(cfg.get("num_workers", 0) or 0):
        parts.append(f"num_workers={int(cfg['num_workers'])}")
    if bool(cfg.get("pin_memory", False)):
        parts.append("pin_memory=True")
    return "".join(f", {p}" for p in parts)


_AUGMENTATIONS: list[tuple[str, str]] = [  # canonical order (applied before ToTensor)
    ("RandomHorizontalFlip", "transforms.RandomHorizontalFlip()"),
    ("RandomVerticalFlip", "transforms.RandomVerticalFlip()"),
    ("Grayscale", "transforms.Grayscale()"),
]


def _compose_transforms(augmentations: list[str], resize: int | None = None) -> tuple[str, str]:
    """(train_transform, eval_transform) Compose expressions. A Resize (if set) is
    deterministic and leads both. Augmentations are train-only, in canonical order
    before ToTensor; eval/val gets Resize + ToTensor so validation isn't perturbed
    by random augmentation."""
    prefix = [f"transforms.Resize(({int(resize)}, {int(resize)}))"] if resize else []
    picked = [expr for name, expr in _AUGMENTATIONS if name in augmentations]
    train = ", ".join([*prefix, *picked, "transforms.ToTensor()"])
    eval_ = ", ".join([*prefix, "transforms.ToTensor()"])
    return f"transforms.Compose([{train}])", f"transforms.Compose([{eval_}])"


def _dataloader_memory(
    cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, namespace: dict | None, n_inputs: int
) -> str:
    """In-memory source. An optionally-picked notebook variable gets the wrapping
    its *type* calls for: a DataLoader passes through, a Dataset is wrapped; a
    tensor/array pick — or no pick at all — falls back to the generic TensorDataset
    path (make_dataloaders(X, y), one X per model input)."""
    from .introspect import variable_kind

    x_var = str(cfg.get("x_var", "") or "").strip()
    kind = variable_kind(x_var, namespace) if x_var else None

    if kind == "dataloader":
        # Already a DataLoader — nothing to build; hand it straight to train().
        return "def make_dataloaders(loader):\n    return loader, None\n"
    if kind == "dataset":
        return (
            "from torch.utils.data import DataLoader\n\n\n"
            f"def make_dataloaders(dataset, *, batch_size={batch_size}):\n"
            f"    return DataLoader(dataset, batch_size=batch_size, shuffle={shuffle}{drop}{common}), None\n"
        )
    # tensors / ndarray / unknown → the TensorDataset wrapping (one X per model input).
    return _dataloader_tensors(cfg, batch_size, shuffle, drop, common, n_inputs)


def _dataloader_tensors(cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, n_inputs: int) -> str:
    """In-memory tensors → a DataLoader over a TensorDataset, with one X arg per
    model input (X for single-input, X0/X1/… for multi). With val_split > 0, a
    disjoint random_split yields a held-out val_loader too."""
    val_split = float(cfg["val_split"])
    xs = ["X"] if n_inputs <= 1 else [f"X{i}" for i in range(n_inputs)]
    x_params = ", ".join(xs)  # make_dataloaders params + TensorDataset args
    lines = [
        "import torch",
        "from torch.utils.data import DataLoader, TensorDataset",
        "",
        "",
    ]
    if val_split > 0.0:
        lines += [
            f"def make_dataloaders({x_params}, y, *, batch_size={batch_size}, val_split={val_split!r}):",
            f"    dataset = TensorDataset({x_params}, y)",
            "    n_val = int(len(dataset) * val_split)",
            "    n_train = len(dataset) - n_val",
            "    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])",
            f"    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle={shuffle}{drop}{common})",
            f"    val_loader = DataLoader(val_ds, batch_size=batch_size{common})",
            "    return train_loader, val_loader",
        ]
    else:
        lines += [
            f"def make_dataloaders({x_params}, y, *, batch_size={batch_size}):",
            f"    dataset = TensorDataset({x_params}, y)",
            f"    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle={shuffle}{drop}{common})",
            "    return train_loader, None",
        ]
    return "\n".join(lines) + "\n"


def _dataloader_torchvision(cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str) -> str:
    """A torchvision dataset → train (train=True) and val (train=False, the test
    split) DataLoaders. Train-only augmentations compose before ToTensor; val gets
    a plain ToTensor."""
    dataset = str(cfg["dataset"])
    root = str(cfg["root"])
    download = bool(cfg["download"])
    train_tf, eval_tf = _compose_transforms(list(cfg.get("augmentations") or []), cfg.get("resize"))

    lines = [
        "from torch.utils.data import DataLoader",
        "from torchvision import datasets, transforms",
        "",
        "",
        f"def make_dataloaders(*, batch_size={batch_size}, root={root!r}):",
    ]
    if train_tf == eval_tf:  # no augmentations — one shared transform
        lines.append(f"    transform = {train_tf}")
        train_arg = eval_arg = "transform"
    else:
        lines += [f"    train_transform = {train_tf}", f"    eval_transform = {eval_tf}"]
        train_arg, eval_arg = "train_transform", "eval_transform"
    lines += [
        f"    train_ds = datasets.{dataset}(root, train=True, download={download}, transform={train_arg})",
        f"    val_ds = datasets.{dataset}(root, train=False, download={download}, transform={eval_arg})",
        f"    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle={shuffle}{drop}{common})",
        f"    val_loader = DataLoader(val_ds, batch_size=batch_size{common})",
        "    return train_loader, val_loader",
    ]
    return "\n".join(lines) + "\n"


def _dataloader_imagefolder(cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str) -> str:
    """A directory of class-subfolders via datasets.ImageFolder. One dataset with a
    deterministic transform (Resize + ToTensor); val_split > 0 carves a held-out
    val_loader via random_split. (Augmentations are torchvision-only — a split
    subset shares one transform, so train-only augmentation can't apply cleanly.)"""
    root = str(cfg["root"])
    val_split = float(cfg.get("val_split", 0.0) or 0.0)
    transform, _ = _compose_transforms([], cfg.get("resize"))  # deterministic; train == eval

    dl_import = "from torch.utils.data import DataLoader, random_split" if val_split > 0.0 \
        else "from torch.utils.data import DataLoader"
    lines = [
        dl_import,
        "from torchvision import datasets, transforms",
        "",
        "",
    ]
    if val_split > 0.0:
        lines += [
            f"def make_dataloaders(*, batch_size={batch_size}, root={root!r}, val_split={val_split!r}):",
            f"    transform = {transform}",
            "    dataset = datasets.ImageFolder(root, transform=transform)",
            "    n_val = int(len(dataset) * val_split)",
            "    n_train = len(dataset) - n_val",
            "    train_ds, val_ds = random_split(dataset, [n_train, n_val])",
            f"    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle={shuffle}{drop}{common})",
            f"    val_loader = DataLoader(val_ds, batch_size=batch_size{common})",
            "    return train_loader, val_loader",
        ]
    else:
        lines += [
            f"def make_dataloaders(*, batch_size={batch_size}, root={root!r}):",
            f"    transform = {transform}",
            "    dataset = datasets.ImageFolder(root, transform=transform)",
            f"    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle={shuffle}{drop}{common})",
            "    return train_loader, None",
        ]
    return "\n".join(lines) + "\n"
