import keyword

import torch
import torch.nn as nn
from .registry import REGISTRY, ModuleEmit, build_module_args
from .schema import Graph, ModelDef, Project


def _name_issues(graph: Graph) -> list[str]:
    """Validate the optional `name` params on Input/Output nodes: each must be a
    usable Python identifier (it becomes a forward() arg or a namedtuple field)
    and unique within its kind. Blank names auto-name later, so they're skipped."""
    issues: list[str] = []
    for kind, label in (("Input", "Input"), ("Output", "Output")):
        seen: set[str] = set()
        for node in graph.nodes:
            if node.type != kind:
                continue
            name = str(node.params.get("name", "") or "").strip()
            if not name:
                continue
            if not name.isidentifier() or keyword.iskeyword(name) or name == "self":
                issues.append(f"{label} name '{name}' is not a valid identifier.")
            elif kind == "Output" and name.startswith("_"):
                # namedtuple rejects leading-underscore field names.
                issues.append(f"Output name '{name}' can't start with an underscore.")
            elif name in seen:
                issues.append(f"Duplicate {label} name '{name}'.")
            else:
                seen.add(name)
    return issues


def build_incoming(graph: Graph) -> dict[str, dict[str, tuple[str, str]]]:
    """node id -> {target_handle: (source_node_id, source_handle)}. One edge per
    input handle. The source handle is kept so a node can wire to a specific
    output pin of a multi-output source (e.g. an LSTM's output vs h_n)."""
    incoming: dict[str, dict[str, tuple[str, str]]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = (edge.source, edge.sourceHandle)
    return incoming


def topo_order(
    graph: Graph, incoming: dict[str, dict[str, tuple[str, str]]]
) -> tuple[list[str], set[str]]:
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
        for src, _handle in incoming.get(node_id, {}).values():
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
    # Multiple Input nodes are allowed — each becomes a forward() argument.
    if n_out == 0:
        issues.append("No Output node — add one to mark the model's result.")
    # Multiple Output nodes are allowed — the model returns a tuple of them.
    issues += _name_issues(graph)
    return issues


def infer_shapes(
    graph: Graph, param_counts: dict[str, dict] | None = None
) -> tuple[dict[tuple[str, str], list[int]], dict[str, str]]:
    """Run meta-tensor shape inference. Returns (shapes, errors): shapes keyed by
    (node id, output pin) so multi-output nodes (e.g. LSTM) get a shape per pin;
    errors keyed by node id.

    Pass ``param_counts`` (an empty dict) to also collect each layer node's
    parameter count and the shapes of its parameter tensors (the count's
    factorization, e.g. Linear → [[128, 784], [128]] = 128×784 + 128) — free
    here, since the real module is instantiated on the meta device anyway."""
    shapes: dict[tuple[str, str], list[int]] = {}
    errors: dict[str, str] = {}
    # Output dtype per (node, pin) — so an Embedding's index input is built as a
    # LongTensor on the meta device rather than the default float.
    dtypes: dict[tuple[str, str], torch.dtype] = {}

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
                shapes[(node_id, "output")] = dims
                dtypes[(node_id, "output")] = torch.long if p.get("dtype") == "long" else torch.float32
                continue

            if not ins:
                errors[node_id] = "no input connected"
                continue

            # Sources as (node, output-pin) keys, in deterministic handle order.
            src_keys = [ins[h] for h in sorted(ins)]
            if any(src in errors for src, _ in src_keys):
                errors[node_id] = "upstream error"
                continue
            if any(key not in shapes for key in src_keys):
                errors[node_id] = "disconnected"
                continue

            with torch.device("meta"):
                if node.type == "Concat":
                    in_shapes = [shapes[k] for k in src_keys]
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
                    shapes[(node_id, "output")] = out
                    dtypes[(node_id, "output")] = dtypes[src_keys[0]]
                    continue

                # Single-input ops. Standard layers (ModuleEmit) are built on the
                # meta device and run; the Output sink preserves the shape.
                input_shape = shapes[src_keys[0]]
                input_dtype = dtypes[src_keys[0]]
                node_def = REGISTRY.get(node.type)
                emit = node_def.emit if node_def else None

                if node.type == "Output":
                    shapes[(node_id, "output")] = list(input_shape)
                    dtypes[(node_id, "output")] = input_dtype

                elif isinstance(emit, ModuleEmit):
                    if emit.min_rank is not None and len(input_shape) < emit.min_rank:
                        msg = emit.rank_msg or f"{emit.cls} expects rank ≥{emit.min_rank}, got {{rank}}"
                        raise ValueError(msg.format(rank=len(input_shape)))
                    # Meta tensors skip dtype checks in forward, so enforce the
                    # integer-index requirement explicitly (otherwise it only
                    # surfaces at runtime).
                    if emit.int_input and input_dtype != torch.long:
                        raise ValueError(
                            f"{emit.cls} expects an integer index input — set the Input's dtype to 'long'"
                        )
                    pos, kw = build_module_args(node_def, p, input_shape)
                    # eval() so only the shape transform runs — no training-time
                    # checks (BatchNorm batch-size / momentum=None .item()) that
                    # are irrelevant to shape and break on meta tensors.
                    module = getattr(nn, emit.cls)(*pos, **kw).eval()
                    if param_counts is not None:
                        tensors = list(module.parameters())
                        param_counts[node_id] = {
                            "count": sum(p.numel() for p in tensors),
                            "terms": [list(p.shape) for p in tensors],
                        }
                    ret = module(torch.empty(input_shape, dtype=input_dtype))
                    # Pull each declared output pin out of the (possibly nested)
                    # return value by its index path.
                    for pin, path in emit.outputs:
                        t = ret
                        for i in path:
                            t = t[i]
                        shapes[(node_id, pin)] = list(t.shape)
                        dtypes[(node_id, pin)] = t.dtype

                else:
                    errors[node_id] = f"unknown node type '{node.type}'"

        except Exception as exc:
            errors[node_id] = str(exc)

    return shapes, errors


def primary_shapes(
    graph: Graph, shapes: dict[tuple[str, str], list[int]]
) -> dict[str, list[int]]:
    """Per-node display shape (the node's first output pin), for the editor's
    node→shape readout — collapsing the per-pin map back to per-node."""
    out: dict[str, list[int]] = {}
    for node in graph.nodes:
        node_def = REGISTRY.get(node.type)
        pin = node_def.outputs[0].name if (node_def and node_def.outputs) else "output"
        if (node.id, pin) in shapes:
            out[node.id] = shapes[(node.id, pin)]
    return out


def _fmt_shape(dims: list[int]) -> str:
    """A link message's shape, batch dim shown as N (matching the canvas badges)."""
    return " × ".join(["N", *(str(d) for d in dims[1:])]) if dims else "?"


def _endpoint_shape(
    model: ModelDef, node_id: str | None, kind: str, shapes: dict[str, list[int]]
) -> tuple[list[int] | None, str | None]:
    """The shape at one end of a link: the named Input/Output node, or the sole
    one when unspecified. Returns (shape, error) — exactly one is set."""
    candidates = [n for n in model.graph.nodes if n.type == kind]
    if node_id is not None:
        node = next((n for n in candidates if n.id == node_id), None)
        if node is None:
            return None, f"{model.name} has no {kind} node for this link"
    elif len(candidates) == 1:
        node = candidates[0]
    elif not candidates:
        return None, f"{model.name} has no {kind} node"
    else:
        return None, f"{model.name} has several {kind} nodes — pick one for the link"
    shape = shapes.get(node.id)
    if shape is None:
        return None, f"{model.name} {kind.lower()} shape is unknown (fix the model first)"
    return shape, None


def link_issues(
    project: Project, model_shapes: dict[str, dict[str, list[int]]]
) -> list[dict]:
    """Shape-check every model link: the source model's Output must match the
    target model's Input. ``model_shapes`` is the per-model primary-shape map
    (``{model_id: {node_id: dims}}``). Returns one result per link:
    ``{id, ok, message}`` — the message reads as evidence on the system canvas
    (``Generator → Discriminator: N × 784``) or the mismatch that breaks it."""
    by_id = {m.id: m for m in project.models}
    out: list[dict] = []
    for link in project.links:
        if link.source_data is not None:
            # Data-node → model wires are shape-checked in a later slice (they
            # need the data node's resolved output shape); skip for now.
            continue
        src = by_id.get(link.source_model)
        tgt = by_id.get(link.target_model)
        if src is None or tgt is None:
            out.append({"id": link.id, "ok": False, "message": "link references a missing model"})
            continue
        src_shape, src_err = _endpoint_shape(src, link.source_pin, "Output", model_shapes.get(src.id, {}))
        tgt_shape, tgt_err = _endpoint_shape(tgt, link.target_input, "Input", model_shapes.get(tgt.id, {}))
        if src_err or tgt_err:
            out.append({"id": link.id, "ok": False, "message": src_err or tgt_err})
        elif src_shape == tgt_shape:
            out.append({"id": link.id, "ok": True, "message": f"{src.name} → {tgt.name}: {_fmt_shape(src_shape)}"})
        else:
            out.append({
                "id": link.id,
                "ok": False,
                "message": f"{src.name} output {_fmt_shape(src_shape)} ≠ {tgt.name} input {_fmt_shape(tgt_shape)}",
            })
    return out


def pin_shapes(
    shapes: dict[tuple[str, str], list[int]]
) -> dict[str, dict[str, list[int]]]:
    """Nest the per-pin shape map as ``{node_id: {pin: dims}}`` — a JSON-friendly
    shape (tuple keys can't serialize) that lets the Inspector show every output
    pin of a multi-output node (e.g. an LSTM's output / h_n / c_n)."""
    out: dict[str, dict[str, list[int]]] = {}
    for (nid, pin), dims in shapes.items():
        out.setdefault(nid, {})[pin] = dims
    return out
