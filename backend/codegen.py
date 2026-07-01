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
    """A self-contained `train(model, X, y)` function from the graph's training
    config (loss/optimizer/hyperparams, validation split, metric). Independent of
    the model architecture — you build the model separately and pass it in along
    with your own data."""
    cfg = {**default_training(), **(graph.training or {})}
    loss = str(cfg["loss"])
    optimizer = str(cfg["optimizer"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    val_split = float(cfg["val_split"])
    metric = str(cfg["metric"])
    device = str(cfg["device"])
    data = str(cfg["data"])

    # Top-1 (argmax) accuracy is only meaningful for classification losses, so
    # gate it on the loss — a regression loss never emits accuracy code.
    track_acc = metric == "accuracy" and loss in ("CrossEntropyLoss", "NLLLoss")
    has_val = val_split > 0.0

    # A multi-input model takes several tensors, so train() accepts a tuple `Xs`
    # and calls model(*batch); a single-input model keeps the plain `X` form.
    incoming = build_incoming(graph)
    node_map = {n.id: n for n in graph.nodes}
    multi = len(model_inputs(graph, incoming, node_map)) > 1

    opt_args = [f"lr={lr!r}"]
    if weight_decay != 0.0:  # omit the default for cleaner code
        opt_args.append(f"weight_decay={weight_decay!r}")
    opt_call = f"torch.optim.{optimizer}(model.parameters(), {', '.join(opt_args)})"

    # DataLoader mode is a distinct loop: iterate the loader (bring-your-own
    # train/val loaders) rather than indexing in-memory tensors. batch_size /
    # val_split don't apply — the loader owns batching and you pass a val_loader.
    if data == "dataloader":
        return _generate_training_dataloader(loss, opt_call, track_acc, multi, epochs, device)

    x_param = "Xs" if multi else "X"
    call = "model(*xb)" if multi else "model(xb)"
    # Val runs full-batch; move its inputs to the device inline (the val set isn't
    # kept resident, unlike the per-batch training moves below).
    val_call = "model(*(x.to(device) for x in X_val))" if multi else "model(X_val.to(device))"
    size0 = "Xs[0].size(0)" if multi else "X.size(0)"

    sig = f"def train(model, {x_param}, y, *, epochs={epochs}, batch_size={batch_size}"
    if has_val:
        sig += f", val_split={val_split!r}"
    sig += f", device={device!r}):"

    lines = ["import torch", "import torch.nn as nn", "", "", sig]
    lines += _device_resolution_lines()

    if has_val:
        lines += [
            f"    n = {size0}",
            "    split = int(n * (1 - val_split))",
            "    perm = torch.randperm(n)",
            "    train_idx, val_idx = perm[:split], perm[split:]",
        ]
        if multi:
            lines += [
                "    X_train = tuple(X[train_idx] for X in Xs)",
                "    X_val = tuple(X[val_idx] for X in Xs)",
                "    y_train, y_val = y[train_idx], y[val_idx]",
            ]
        else:
            lines += [
                "    X_train, y_train = X[train_idx], y[train_idx]",
                "    X_val, y_val = X[val_idx], y[val_idx]",
            ]
    else:
        lines.append(f"    X_train, y_train = {x_param}, y")

    lines += [
        f"    loss_fn = nn.{loss}()",
        f"    opt = {opt_call}",
        f"    n_train = {'X_train[0]' if multi else 'X_train'}.size(0)",
        "    for epoch in range(epochs):",
        "        model.train()",
        "        order = torch.randperm(n_train)",
        "        running = 0.0",
    ]
    if track_acc:
        lines.append("        correct = 0")
    lines.append("        for i in range(0, n_train, batch_size):")
    lines.append("            idx = order[i:i + batch_size]")
    if multi:
        lines += [
            "            xb = tuple(X[idx].to(device) for X in X_train)",
            "            yb = y_train[idx].to(device)",
        ]
    else:
        lines.append("            xb, yb = X_train[idx].to(device), y_train[idx].to(device)")
    lines += [
        "            opt.zero_grad()",
        f"            out = {call}",
        "            loss = loss_fn(out, yb)",
        "            loss.backward()",
        "            opt.step()",
        f"            running += loss.item() * {'yb' if multi else 'xb'}.size(0)",
    ]
    if track_acc:
        lines.append("            correct += (out.argmax(dim=-1) == yb).sum().item()")
    lines.append("        train_loss = running / n_train")
    if track_acc:
        lines.append("        train_acc = correct / n_train")

    if has_val:
        lines += [
            "        model.eval()",
            "        with torch.no_grad():",
            f"            val_out = {val_call}",
            "            yv = y_val.to(device)",
            "            val_loss = loss_fn(val_out, yv).item()",
        ]
        if track_acc:
            lines.append(
                "            val_acc = (val_out.argmax(dim=-1) == yv).float().mean().item()"
            )

    # Assemble the per-epoch report. msg holds literal f-string fields; the outer
    # f-string only substitutes msg, so the braces survive into the generated line.
    msg = "epoch {epoch + 1}/{epochs}  loss {train_loss:.4f}"
    if track_acc:
        msg += " acc {train_acc:.3f}"
    if has_val:
        msg += "  val_loss {val_loss:.4f}"
        if track_acc:
            msg += " val_acc {val_acc:.3f}"
    lines.append(f'        print(f"{msg}")')
    lines.append("    return model")

    return "\n".join(lines) + "\n"


def generate_dataloader(graph: Graph) -> str:
    """A `make_dataloaders()` helper from the Data panel's config, returning
    (train_loader, val_loader). It pairs with the DataLoader training mode:
    `train_loader, val_loader = make_dataloaders(...)` then
    `train(model, train_loader, val_loader=val_loader)`."""
    cfg = {**default_data(), **(graph.data or {})}
    source = str(cfg["source"])
    batch_size = int(cfg["batch_size"])
    shuffle = bool(cfg["shuffle"])
    # Only the train loader drops a ragged batch; omitted when off for clean code.
    drop = ", drop_last=True" if bool(cfg["drop_last"]) else ""
    if source == "torchvision":
        return _dataloader_torchvision(cfg, batch_size, shuffle, drop)
    return _dataloader_tensors(cfg, batch_size, shuffle, drop)


def _dataloader_tensors(cfg: dict, batch_size: int, shuffle: bool, drop: str) -> str:
    """In-memory tensors → a DataLoader over a TensorDataset. With val_split > 0,
    a disjoint random_split yields a held-out val_loader too."""
    val_split = float(cfg["val_split"])
    lines = [
        "import torch",
        "from torch.utils.data import DataLoader, TensorDataset",
        "",
        "",
    ]
    if val_split > 0.0:
        lines += [
            f"def make_dataloaders(X, y, *, batch_size={batch_size}, val_split={val_split!r}):",
            "    dataset = TensorDataset(X, y)",
            "    n_val = int(len(dataset) * val_split)",
            "    n_train = len(dataset) - n_val",
            "    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])",
            f"    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle={shuffle}{drop})",
            "    val_loader = DataLoader(val_ds, batch_size=batch_size)",
            "    return train_loader, val_loader",
        ]
    else:
        lines += [
            f"def make_dataloaders(X, y, *, batch_size={batch_size}):",
            "    dataset = TensorDataset(X, y)",
            f"    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle={shuffle}{drop})",
            "    return train_loader, None",
        ]
    return "\n".join(lines) + "\n"


def _dataloader_torchvision(cfg: dict, batch_size: int, shuffle: bool, drop: str) -> str:
    """A torchvision dataset → train (train=True) and val (train=False, the test
    split) DataLoaders. Slice 1 uses a plain ToTensor transform."""
    dataset = str(cfg["dataset"])
    root = str(cfg["root"])
    download = bool(cfg["download"])
    lines = [
        "from torch.utils.data import DataLoader",
        "from torchvision import datasets, transforms",
        "",
        "",
        f"def make_dataloaders(*, batch_size={batch_size}, root={root!r}):",
        "    transform = transforms.ToTensor()",
        f"    train_ds = datasets.{dataset}(root, train=True, download={download}, transform=transform)",
        f"    val_ds = datasets.{dataset}(root, train=False, download={download}, transform=transform)",
        f"    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle={shuffle}{drop})",
        "    val_loader = DataLoader(val_ds, batch_size=batch_size)",
        "    return train_loader, val_loader",
    ]
    return "\n".join(lines) + "\n"


def _generate_training_dataloader(
    loss: str, opt_call: str, track_acc: bool, multi: bool, epochs: int, device: str
) -> str:
    """DataLoader variant of train(): iterate a torch DataLoader (yielding
    (inputs…, target) batches) instead of indexing in-memory tensors. An optional
    val_loader runs validation. Handles single- and multi-input models uniformly
    via `*xb, yb = batch` — one trailing target, the rest are model inputs."""
    if multi:
        unpack, to_dev, call = "*xb, yb = batch", "xb = [t.to(device) for t in xb]", "model(*xb)"
    else:
        unpack, to_dev, call = "xb, yb = batch", "xb = xb.to(device)", "model(xb)"

    lines = ["import torch", "import torch.nn as nn", "", ""]
    lines.append(f"def train(model, loader, *, epochs={epochs}, val_loader=None, device={device!r}):")
    lines += _device_resolution_lines()
    lines += [
        f"    loss_fn = nn.{loss}()",
        f"    opt = {opt_call}",
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

    lines.append("        print(msg)")
    lines.append("    return model")
    return "\n".join(lines) + "\n"
