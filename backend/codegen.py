from .schema import Graph
from .inference import infer_shapes, build_incoming, topo_order
from .registry import REGISTRY, ModuleEmit, render_module_args


def generate_module(graph: Graph) -> str:
    shapes, errors = infer_shapes(graph)
    if errors:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise ValueError(f"Graph has errors — {detail}")

    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    order, _ = topo_order(graph, incoming)

    inputs = [nid for nid in order if node_map[nid].type == "Input"]
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
