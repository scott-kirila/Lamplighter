import torch
import torch.nn as nn
from .schema import Graph


def build_incoming(graph: Graph) -> dict[str, dict[str, str]]:
    """node id -> {target_handle: source_node_id}. One edge per input handle."""
    incoming: dict[str, dict[str, str]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = edge.source
    return incoming


def topo_order(graph: Graph, incoming: dict[str, dict[str, str]]) -> tuple[list[str], set[str]]:
    """DFS topological order. Returns (order, cyclic_node_ids)."""
    visited: set[str] = set()
    cyclic: set[str] = set()
    order: list[str] = []

    def visit(node_id: str, stack: frozenset[str] = frozenset()) -> None:
        if node_id in visited:
            return
        if node_id in stack:
            cyclic.add(node_id)
            return
        for src in incoming.get(node_id, {}).values():
            visit(src, stack | {node_id})
        visited.add(node_id)
        order.append(node_id)

    for node in graph.nodes:
        visit(node.id)
    return order, cyclic


def infer_shapes(graph: Graph) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Run meta-tensor shape inference. Returns (shapes, errors) keyed by node id."""
    shapes: dict[str, list[int]] = {}
    errors: dict[str, str] = {}

    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    order, cyclic = topo_order(graph, incoming)
    for nid in cyclic:
        errors[nid] = "cycle detected"

    for node_id in order:
        if node_id in errors:
            continue
        node = node_map[node_id]
        p = node.params
        ins = incoming.get(node_id, {})

        try:
            if node.type == "Input":
                raw = str(p.get("shape", "1, 784"))
                dims = [int(tok) for tok in raw.split(",") if tok.strip() != ""]
                if not dims:
                    raise ValueError(f"invalid shape '{raw}'")
                shapes[node_id] = dims
                continue

            if not ins:
                errors[node_id] = "no input connected"
                continue

            # Sources resolved in deterministic handle order (in0, in1, …)
            src_ids = [ins[h] for h in sorted(ins)]
            if any(s in errors for s in src_ids):
                errors[node_id] = "upstream error"
                continue
            if any(s not in shapes for s in src_ids):
                errors[node_id] = "disconnected"
                continue

            with torch.device("meta"):
                if node.type == "Concat":
                    in_shapes = [shapes[s] for s in src_ids]
                    if len(in_shapes) < 2:
                        raise ValueError("Concat needs ≥2 inputs")
                    rank = len(in_shapes[0])
                    if any(len(s) != rank for s in in_shapes):
                        raise ValueError("rank mismatch between inputs")
                    dim = int(p.get("dim", 1))
                    d = dim if dim >= 0 else rank + dim
                    if not (0 <= d < rank):
                        raise ValueError(f"dim {dim} out of range for rank {rank}")
                    for ax in range(rank):
                        if ax == d:
                            continue
                        if len({s[ax] for s in in_shapes}) != 1:
                            sizes = [s[ax] for s in in_shapes]
                            raise ValueError(f"size mismatch on dim {ax}: {sizes}")
                    out = list(in_shapes[0])
                    out[d] = sum(s[d] for s in in_shapes)
                    shapes[node_id] = out
                    continue

                # Single-input ops
                input_shape = shapes[src_ids[0]]
                x = torch.empty(input_shape)

                if node.type in ("ReLU", "Sigmoid", "Tanh", "Dropout", "Output"):
                    shapes[node_id] = list(x.shape)

                elif node.type == "Linear":
                    in_f = input_shape[-1]
                    out_f = int(p.get("out_features", 128))
                    bias = bool(p.get("bias", True))
                    shapes[node_id] = list(nn.Linear(in_f, out_f, bias=bias)(x).shape)

                elif node.type == "Conv2d":
                    if len(input_shape) < 4:
                        raise ValueError(f"Conv2d expects 4D input (B,C,H,W), got {len(input_shape)}D")
                    in_ch = input_shape[1]
                    out_ch = int(p.get("out_channels", 32))
                    ks = int(p.get("kernel_size", 3))
                    st = int(p.get("stride", 1))
                    pad = int(p.get("padding", 0))
                    shapes[node_id] = list(nn.Conv2d(in_ch, out_ch, ks, st, pad)(x).shape)

                elif node.type == "Flatten":
                    start = int(p.get("start_dim", 1))
                    shapes[node_id] = list(nn.Flatten(start_dim=start)(x).shape)

                elif node.type == "BatchNorm1d":
                    num_f = input_shape[-1]
                    shapes[node_id] = list(nn.BatchNorm1d(num_f)(x).shape)

                else:
                    errors[node_id] = f"unknown node type '{node.type}'"

        except Exception as exc:
            errors[node_id] = str(exc)

    return shapes, errors
