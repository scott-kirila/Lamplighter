from .schema import Graph
from .inference import infer_shapes, build_incoming, topo_order
from .registry import REGISTRY, ModuleEmit, default_training, render_module_args


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

    # Top-1 (argmax) accuracy is only meaningful for classification losses, so
    # gate it on the loss — a regression loss never emits accuracy code.
    track_acc = metric == "accuracy" and loss in ("CrossEntropyLoss", "NLLLoss")
    has_val = val_split > 0.0

    # A multi-input model takes several tensors, so train() accepts a tuple `Xs`
    # and calls model(*batch); a single-input model keeps the plain `X` form.
    incoming = build_incoming(graph)
    node_map = {n.id: n for n in graph.nodes}
    multi = len(model_inputs(graph, incoming, node_map)) > 1
    x_param = "Xs" if multi else "X"
    call = "model(*xb)" if multi else "model(xb)"
    val_call = "model(*X_val)" if multi else "model(X_val)"
    size0 = "Xs[0].size(0)" if multi else "X.size(0)"

    opt_args = [f"lr={lr!r}"]
    if weight_decay != 0.0:  # omit the default for cleaner code
        opt_args.append(f"weight_decay={weight_decay!r}")
    opt_call = f"torch.optim.{optimizer}(model.parameters(), {', '.join(opt_args)})"

    sig = f"def train(model, {x_param}, y, *, epochs={epochs}, batch_size={batch_size}"
    if has_val:
        sig += f", val_split={val_split!r}"
    sig += "):"

    lines = ["import torch", "import torch.nn as nn", "", "", sig]

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
            "            xb = tuple(X[idx] for X in X_train)",
            "            yb = y_train[idx]",
        ]
    else:
        lines.append("            xb, yb = X_train[idx], y_train[idx]")
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
            "            val_loss = loss_fn(val_out, y_val).item()",
        ]
        if track_acc:
            lines.append(
                "            val_acc = (val_out.argmax(dim=-1) == y_val).float().mean().item()"
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
