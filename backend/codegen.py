from .schema import Graph
from .inference import infer_shapes, build_incoming, topo_order
from .registry import REGISTRY, ModuleEmit, render_module_args


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
