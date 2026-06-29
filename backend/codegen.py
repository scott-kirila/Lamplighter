from .schema import Graph
from .inference import infer_shapes, build_incoming, topo_order
from .registry import REGISTRY, ModuleEmit, default_training, render_module_args


def generate_module(graph: Graph) -> str:
    shapes, errors = infer_shapes(graph)

    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    order, _ = topo_order(graph, incoming)

    outputs = [nid for nid in order if node_map[nid].type == "Output"]
    if len(outputs) != 1:
        raise ValueError(f"expected exactly 1 Output node, found {len(outputs)}")

    # The model is the subgraph that feeds the Output — the nodes backward-
    # reachable from it. Stray/disconnected nodes (and any errors they carry) are
    # ignored, so a scratch node on the canvas doesn't break codegen.
    live: set[str] = set()
    stack = [outputs[0]]
    while stack:
        nid = stack.pop()
        if nid in live:
            continue
        live.add(nid)
        stack.extend(incoming.get(nid, {}).values())

    live_errors = {k: v for k, v in errors.items() if k in live}
    if live_errors:
        detail = "; ".join(f"{k}: {v}" for k, v in live_errors.items())
        raise ValueError(f"Graph has errors — {detail}")

    inputs = [nid for nid in order if node_map[nid].type == "Input" and nid in live]
    if len(inputs) != 1:
        raise ValueError(f"expected exactly 1 Input node, found {len(inputs)}")

    # Each node's output gets an SSA variable; the Input maps to the forward arg.
    var: dict[str, str] = {inputs[0]: "x"}
    output_var = "x"
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
            output_var = var[next(iter(incoming[nid].values()))]
            continue

        v = f"t{counter}"
        counter += 1
        var[nid] = v

        if t == "Concat":
            handles = sorted(incoming[nid])
            args = ", ".join(var[incoming[nid][h]] for h in handles)
            dim = int(p.get("dim", 1))
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
            fwd_lines.append(f"{v} = self.layer_{midx}({sv(nid)})")
            midx += 1

    parts = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class GeneratedModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    parts += ["        " + line for line in (init_lines or ["pass"])]
    parts += ["", "    def forward(self, x):"]
    parts += ["        " + line for line in (fwd_lines or ["pass"])]
    parts.append(f"        return {output_var}")

    return "\n".join(parts) + "\n"


def generate_training(graph: Graph) -> str:
    """A self-contained `train(model, X, y)` function from the graph's training
    config (loss/optimizer/hyperparams). Independent of the model architecture —
    you build the model separately and pass it in along with your own data."""
    cfg = {**default_training(), **(graph.training or {})}
    loss = str(cfg["loss"])
    optimizer = str(cfg["optimizer"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])

    opt_args = [f"lr={lr!r}"]
    if weight_decay != 0.0:  # omit the default for cleaner code
        opt_args.append(f"weight_decay={weight_decay!r}")
    opt_call = f"torch.optim.{optimizer}(model.parameters(), {', '.join(opt_args)})"

    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        f"def train(model, X, y, *, epochs={epochs}, batch_size={batch_size}):",
        "    model.train()",
        f"    loss_fn = nn.{loss}()",
        f"    opt = {opt_call}",
        "    n = X.size(0)",
        "    for epoch in range(epochs):",
        "        perm = torch.randperm(n)",
        "        running = 0.0",
        "        for i in range(0, n, batch_size):",
        "            idx = perm[i:i + batch_size]",
        "            opt.zero_grad()",
        "            loss = loss_fn(model(X[idx]), y[idx])",
        "            loss.backward()",
        "            opt.step()",
        "            running += loss.item() * idx.size(0)",
        '        print(f"epoch {epoch + 1}/{epochs}  loss {running / n:.4f}")',
        "    return model",
    ]
    return "\n".join(lines) + "\n"
