import torch
import torch.nn as nn
from .registry import REGISTRY, ModuleEmit, build_module_args
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


def graph_issues(graph: Graph) -> list[str]:
    """Graph-level validation independent of shape inference: the presence and
    count of the IO nodes codegen requires. Returned as plain messages (not keyed
    to a node) for display as a banner. Empty graphs are left alone — a blank
    canvas shouldn't nag.
    """
    if not graph.nodes:
        return []
    issues: list[str] = []
    n_in = sum(1 for n in graph.nodes if n.type == "Input")
    n_out = sum(1 for n in graph.nodes if n.type == "Output")
    if n_in == 0:
        issues.append("No Input node — add one to define the model's input.")
    elif n_in > 1:
        issues.append(f"{n_in} Input nodes — only one is supported.")
    if n_out == 0:
        issues.append("No Output node — add one to mark the model's result.")
    elif n_out > 1:
        issues.append(f"{n_out} Output nodes — only one is supported.")
    return issues


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

                # Single-input ops. Standard layers (ModuleEmit) are built on the
                # meta device and run; the Output sink preserves the shape.
                input_shape = shapes[src_ids[0]]
                node_def = REGISTRY.get(node.type)
                emit = node_def.emit if node_def else None

                if node.type == "Output":
                    shapes[node_id] = list(input_shape)

                elif isinstance(emit, ModuleEmit):
                    if emit.min_rank is not None and len(input_shape) < emit.min_rank:
                        msg = emit.rank_msg or f"{emit.cls} expects rank ≥{emit.min_rank}, got {{rank}}"
                        raise ValueError(msg.format(rank=len(input_shape)))
                    pos, kw = build_module_args(node_def, p, input_shape)
                    # eval() so only the shape transform runs — no training-time
                    # checks (BatchNorm batch-size / momentum=None .item()) that
                    # are irrelevant to shape and break on meta tensors.
                    module = getattr(nn, emit.cls)(*pos, **kw).eval()
                    shapes[node_id] = list(module(torch.empty(input_shape)).shape)

                else:
                    errors[node_id] = f"unknown node type '{node.type}'"

        except Exception as exc:
            errors[node_id] = str(exc)

    return shapes, errors
