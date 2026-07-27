"""Import an existing ``nn.Module`` onto the canvas.

Traces a model with ``torch.fx``, walks the graph, and emits the same
``schema.Graph`` the editor produces by hand — so an imported model flows
through the *unchanged* engines: ``infer_shapes`` sees real layers on the meta
device, ``generate_module`` renders idiomatic source, and the run button trains
it. That last part is what a static viewer (Netron, torchview) structurally
cannot do, and it is the whole point of importing here rather than just drawing.

The one law this obeys: it produces registry DATA and touches no engine. A layer
becomes a node type because the registry already knows that layer; the extractor
reads each param generically via its ``ParamDef``. Nothing here branches on a
specific class.

Fidelity is not best-effort. A value this can't express faithfully is not
approximated — it forces an ``Opaque`` node (see the gate), because a
verification tool that draws a *wrong* picture is worse than one that admits a
gap. This module builds the graph; :func:`fidelity_gate` (imported lazily) makes
the trust judgement.

Traps that cost the prototype real time, encoded here so they don't recur:

* The Input node's ``shape`` includes the batch dim — ``infer_shapes`` reads it
  raw, and per-sample dims produce cascading "expected 4D, got 3D" errors.
* ``torch.fx`` wraps ``torch.cat``'s inputs in a *list*, so a naive
  "args that are Nodes" scan finds zero sources; list/tuple args are flattened.
* A reused module instance (resnet shares ReLUs) is fine when stateless, but a
  reused *parametrized* module would be silently untied into two members — that
  is refused, not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import REGISTRY, ModuleEmit
from .schema import Graph, GraphEdge, GraphNode, NodePosition


class ImportError_(Exception):
    """Import couldn't produce a faithful graph. The message is user-facing and
    names the cause (a trace failure, a tied weight) — error quality is a
    measured strength of this product and this is a first-contact surface."""


# nn class name -> registry key, for the 34 layers the registry already covers.
# Built from the registry so a new ModuleEmit node is importable for free.
_CLS_TO_KEY: dict[str, str] = {
    d.emit.cls: key for key, d in REGISTRY.items() if isinstance(d.emit, ModuleEmit)
}

# Constructor args that never affect the emitted module's structure or weights,
# so a difference in them is not infidelity — the fidelity gate ignores these.
IGNORED_CTOR_ARGS = frozenset({"device", "dtype", "inplace"})


class _Unexpressible(Exception):
    """A functional op carries call arguments the registry node can't represent
    faithfully (a mean over a list of dims, a flatten with a bounded end_dim, an
    add with a non-unit alpha). Signals "make this Opaque" rather than silently
    emitting the node's default and drawing a wrong picture — the exact class of
    bug the fidelity guarantee exists to prevent."""


@dataclass
class _FuncSpec:
    """A functional op (torch.add / a tensor method / F.relu) mapped onto a
    registry node. ``pins`` is how many inputs it consumes (edges); ``lift``
    turns the fx call args into node params, or raises ``_Unexpressible`` when
    the call can't be represented — every op MUST have one, so a dropped
    argument can never masquerade as a faithful import."""
    node_type: str
    pins: int
    lift: Any  # (args, kwargs) -> dict, or raises _Unexpressible


def _lift_cat(args, kwargs) -> dict:
    # torch.cat(tensors, dim) — dim may be positional (args[1]) or keyword.
    dim = kwargs.get("dim")
    if dim is None and len(args) > 1 and isinstance(args[1], int):
        dim = args[1]
    return {"dim": int(dim) if dim is not None else 0}


def _lift_add(args, kwargs) -> dict:
    # x + alpha*y — the Add node is a plain sum, so a non-unit alpha changes the
    # result and can't be expressed.
    if kwargs.get("alpha", 1) != 1:
        raise _Unexpressible()
    return {}


def _lift_relu(_args, _kwargs) -> dict:
    return {}  # inplace/out don't change the value


def _lift_mean(args, kwargs) -> dict:
    # The Mean node reduces ONE integer dim. A list of dims (global average
    # pooling's x.mean([2,3])) or an all-dims reduction can't be expressed.
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
    if not isinstance(dim, int):
        raise _Unexpressible()
    return {"dim": int(dim), "keepdim": bool(keepdim)}


def _lift_flatten(args, kwargs) -> dict:
    # The Flatten node flattens start_dim..end. A bounded end_dim (torch.flatten
    # (x, 1, 2)) isn't expressible.
    start = kwargs.get("start_dim", args[1] if len(args) > 1 else 1)
    end = kwargs.get("end_dim", args[2] if len(args) > 2 else -1)
    if int(end) != -1:
        raise _Unexpressible()
    return {"start_dim": int(start)}


# call_function targets and call_method names the registry can represent. Kept
# small and explicit: an unknown function is not guessed, it goes Opaque.
_FUNCS: dict[Any, _FuncSpec] = {
    "add": _FuncSpec("Add", 2, _lift_add),
    "cat": _FuncSpec("Concat", 2, _lift_cat),
    "concat": _FuncSpec("Concat", 2, _lift_cat),
    "flatten": _FuncSpec("Flatten", 1, _lift_flatten),
    "relu": _FuncSpec("ReLU", 1, _lift_relu),
    "mean": _FuncSpec("Mean", 1, _lift_mean),
}


def _func_key(target) -> str:
    """A stable name for an fx call target (a builtin, a torch function, or a
    method-name string) so it can key _FUNCS."""
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def _coerce(value: Any, ptype: str) -> Any:
    """A live module attribute, coerced to the JSON-able form a ParamDef stores.

    The mirror of registry._cast (which goes the other way, params -> ctor
    args). The subtle ones: a Conv's ``bias`` attribute is a Parameter or None,
    so a bool param reads presence; torch stores ``dilation`` as ``(1, 1)`` where
    the default is ``1``, so a tuple param collapses a uniform tuple back to the
    scalar — without which every conv in every model reads as non-default and
    the gate flags it infidel (a false positive the prototype hit head-on).
    """
    if ptype == "bool":
        # A bias-like attribute is a Parameter (on) or None (off); a Parameter's
        # own truth value is ambiguous, so presence IS the bool.
        import torch

        if isinstance(value, torch.Tensor):
            return True
        return value is not None and bool(value)
    if ptype == "tuple":
        if isinstance(value, (tuple, list)):
            vals = list(value)
            if vals and all(v == vals[0] for v in vals):
                return vals[0]  # (3, 3) -> 3 ; (1, 1) -> 1
            return vals
        return value
    if value is None:
        return None  # an optional param left unset (e.g. BatchNorm momentum=None)
    if ptype == "int":
        return int(value)
    if ptype == "float":
        return float(value)
    if ptype in ("enum", "string"):
        return str(value)
    return value


def _extract_params(module, node_type: str) -> dict[str, Any]:
    """Read a live module's params generically, one ``getattr`` per ParamDef.

    Only params the emit actually consumes are read (derived args like a conv's
    in_channels come from the wiring, not from here — which is why the shapes
    reconcile). A param the module doesn't carry is simply left at its default.
    """
    node_def = REGISTRY[node_type]
    emit = node_def.emit
    assert isinstance(emit, ModuleEmit)
    consumed = set(emit.kw_params)
    for arg in emit.pos:
        if isinstance(arg, str):
            consumed.add(arg)
    out: dict[str, Any] = {}
    pdefs = {p.name: p for p in node_def.params}
    for name in consumed:
        pd = pdefs.get(name)
        if pd is None or not hasattr(module, name):
            continue
        out[name] = _coerce(getattr(module, name), pd.type)
    return out


def _flatten_node_args(args) -> list:
    """Every fx Node reachable through an args tuple, INCLUDING inside a list or
    tuple arg — which is how torch.cat/torch.stack pass their tensors. A plain
    "isinstance(a, Node)" scan misses them and the op reports no inputs."""
    import torch.fx as fx

    found = []
    for a in args:
        if isinstance(a, fx.Node):
            found.append(a)
        elif isinstance(a, (list, tuple)):
            found.extend(x for x in a if isinstance(x, fx.Node))
    return found


def _input_pins(node_type: str) -> list[str]:
    """The input pin names for a node type, in the order fx args map onto them.
    Codegen reads multi-input nodes via sorted(incoming), and the pins here
    (in0/in1, or a lone 'input') sort into that same order."""
    return [p.name for p in REGISTRY[node_type].inputs]


def trace(model, input_shape: tuple[int, ...], input_is_int: bool | None = None) -> dict[str, Any]:
    """fx-trace ``model`` and build a graph description.

    Returns a dict with the built :class:`Graph`, the observed per-node output
    shapes (from ShapeProp on the meta device), the ordered source state_dict
    keys (for weight transfer), and per-node fidelity findings. Raises
    :class:`ImportError_` with torch's own message when the model can't be
    traced (data-dependent control flow — including ``nn.LSTM``).
    """
    import torch.fx as fx

    if len(input_shape) < 2:
        raise ImportError_(
            f"input_shape {tuple(input_shape)} needs a batch dim first, e.g. "
            f"(1, {', '.join(map(str, input_shape))})"
        )

    try:
        gm: fx.GraphModule = fx.symbolic_trace(model)
    except Exception as exc:  # TraceError, and anything torch raises mid-trace
        raise ImportError_(
            f"torch.fx couldn't trace this model — usually data-dependent "
            f"control flow (an if/loop over tensor values, which includes "
            f"nn.LSTM/GRU/RNN). torch said:\n\n  {type(exc).__name__}: {exc}\n\n"
            f"You can still build recurrent models on the canvas directly."
        ) from exc

    modules = dict(gm.named_modules())
    # A real example batch is authoritative about its own dtype; with only a
    # shape to go on, ask the graph what the first layer requires.
    wants_int = _wants_integer_input(gm, modules) if input_is_int is None else input_is_int

    # Refuse a reused PARAMETRIZED module before doing anything else: it would
    # become two independent members and silently untie shared weights.
    seen_targets: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            seen_targets[node.target] = seen_targets.get(node.target, 0) + 1
    for target, count in seen_targets.items():
        if count > 1 and any(True for _ in modules[target].parameters(recurse=False)):
            raise ImportError_(
                f"the module '{target}' is used {count} times and has its own "
                f"weights — importing would split it into {count} independent "
                f"copies and untie those shared weights. Tied-weight models "
                f"aren't supported yet."
            )

    from .importer_gate import assess_module  # lazy: gate imports back here

    fx_to_id: dict[Any, str] = {}
    node_target: dict[str, str] = {}  # module node id -> its source fx target (qualified name)
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    findings: list[dict[str, Any]] = []
    counter = 0

    def new_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}{counter}"

    def add_edge(src_id: str, dst_id: str, dst_pin: str, src_pin: str = "output") -> None:
        edges.append(GraphEdge(
            id=f"{src_id}->{dst_id}:{dst_pin}",
            source=src_id, sourceHandle=src_pin, target=dst_id, targetHandle=dst_pin,
        ))

    def wire(dst_id: str, node_type: str, fx_args) -> None:
        """Connect an op/module node's inputs, in fx arg order onto its pins."""
        sources = _flatten_node_args(fx_args)
        pins = _input_pins(node_type)
        for i, src in enumerate(sources[: len(pins)]):
            src_id = fx_to_id.get(src)
            if src_id is not None:
                add_edge(src_id, dst_id, pins[i])

    def make_opaque(label: str, reason: str, fx_args) -> str:
        nid = new_id("op")
        nodes.append(GraphNode(
            id=nid, type="Opaque", position=NodePosition(x=0, y=0),
            params={"label": label, "summary": reason, "out_shape": ""},
        ))
        findings.append({"id": nid, "kind": "opaque", "label": label, "reason": reason})
        wire(nid, "Opaque", fx_args)
        return nid

    def overflows_pins(node_type: str, fx_args) -> bool:
        """More source inputs than the node has pins — an unrepresentable fan-in
        (a concat of 3+ branches into a 2-pin Concat). Drawing it would silently
        drop branches, so it goes Opaque."""
        return len(_flatten_node_args(fx_args)) > len(_input_pins(node_type))

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            nid = new_id("input")
            nodes.append(GraphNode(
                id=nid, type="Input", position=NodePosition(x=0, y=0),
                params={"shape": ", ".join(map(str, input_shape)),
                        "dtype": "long" if wants_int else "float", "name": ""},
            ))
            fx_to_id[node] = nid

        elif node.op == "call_module":
            module = modules[node.target]
            cls = type(module).__name__
            key = _CLS_TO_KEY.get(cls)
            verdict = assess_module(module, key)
            if verdict.opaque or key is None:
                fx_to_id[node] = make_opaque(cls, verdict.reason or cls, node.args)
            else:
                nid = new_id("mod")
                nodes.append(GraphNode(
                    id=nid, type=key, position=NodePosition(x=0, y=0),
                    params=_extract_params(module, key),
                ))
                node_target[nid] = node.target  # provenance, for order-correct seeding
                fx_to_id[node] = nid
                wire(nid, key, node.args)

        elif node.op in ("call_function", "call_method"):
            label = _func_key(node.target)
            spec = _FUNCS.get(label)
            if spec is None:
                fx_to_id[node] = make_opaque(label, f"no registry node for '{label}'", node.args)
                continue
            if overflows_pins(spec.node_type, node.args):
                fx_to_id[node] = make_opaque(
                    label,
                    f"{label} combines {len(_flatten_node_args(node.args))} inputs but "
                    f"the {spec.node_type} node has {len(_input_pins(spec.node_type))}",
                    node.args,
                )
                continue
            try:
                params = spec.lift(node.args, node.kwargs)
            except _Unexpressible:
                fx_to_id[node] = make_opaque(
                    label, f"{label} uses arguments the {spec.node_type} node can't carry", node.args
                )
                continue
            nid = new_id("op")
            nodes.append(GraphNode(
                id=nid, type=spec.node_type, position=NodePosition(x=0, y=0), params=params,
            ))
            fx_to_id[node] = nid
            wire(nid, spec.node_type, node.args)

        elif node.op == "output":
            nid = new_id("output")
            nodes.append(GraphNode(
                id=nid, type="Output", position=NodePosition(x=0, y=0), params={"name": ""},
            ))
            fx_to_id[node] = nid
            wire(nid, "Output", node.args)

    graph = Graph(nodes=nodes, edges=edges)
    observed = _observed_shapes(gm, input_shape, fx_to_id, wants_int)
    _fill_opaque_shapes(nodes, findings, observed)
    layout(graph)

    opaque_count = sum(1 for f in findings if f["kind"] == "opaque")
    # A whole-model refusal above a threshold. A transformer traces to mostly
    # non-tensor plumbing (getitem/getattr/_assert/eq) — vit_b_16 is 45% opaque
    # — and rendering a hundred labelled holes is worse than one honest
    # sentence. 30% is the line between "a CNN with a few gaps to fill in"
    # (mobilenet's Hardswish is ~23%) and "the wrong shape for a dataflow
    # canvas". The floor of 8 keeps a tiny model with one unmapped op an
    # import-with-a-hole rather than a refusal.
    refused = opaque_count >= max(8, 0.3 * len(nodes))
    reason = ""
    if refused:
        kinds = sorted({f["label"] for f in findings if f["kind"] == "opaque"})
        reason = (
            f"{opaque_count} of {len(nodes)} nodes are operations this canvas "
            f"can't represent ({', '.join(kinds[:5])}"
            f"{', …' if len(kinds) > 5 else ''}). This is usually a transformer "
            f"or a model whose forward() does tensor bookkeeping between layers — "
            f"the shape a dataflow graph can't capture. Nothing was imported."
        )

    # Weights in the order the GENERATED module will expect them, not the source
    # model's registration order. seed_from_weights zips these against the
    # generated state_dict positionally, so they must follow the graph's own
    # layer order — a model whose submodules are registered in one order but
    # applied in another (a ModuleList used in reverse) would otherwise be
    # mis-seeded silently. Built per-node from each layer's source module.
    state_values, state_keys = _weights_by_graph_order(model, graph, node_target)

    return {
        "graph": graph,
        "observed_shapes": observed,
        "state_keys": state_keys,
        "state_values": state_values,
        "findings": findings,
        "opaque_count": opaque_count,
        "refused": refused,
        "refused_reason": reason,
    }


def _weights_by_graph_order(model, graph: Graph, node_target: dict[str, str]):
    """(values, keys) for the model's weights, ordered to match the generated
    module's ``layer_0..layer_N`` state_dict — each layer contributing its
    source submodule's parameters and buffers, in the order codegen names the
    layers. Skips a graph with holes (an Opaque node means codegen refuses
    anyway, and get_submodule would be undefined)."""
    from .codegen import layer_nodes

    values: list = []
    keys: list[str] = []
    for ln in layer_nodes(graph):
        target = node_target.get(ln.node_id)
        if target is None:
            continue  # a functional/op layer with no source module
        submodule = model.get_submodule(target)
        for name, tensor in submodule.state_dict().items():
            values.append(tensor)
            keys.append(f"{target}.{name}" if target else name)
    return values, keys


def _wants_integer_input(gm, modules) -> bool:
    """Does the traced input feed a layer that INDEXES with it?

    ``nn.Embedding`` takes token ids, so a language model's Input is ``long``,
    not ``float``. Stamping every imported Input as float had two costs: the
    pre-flight panel reported the user's real token ids as the wrong dtype (a
    false error on exactly the models the importer was extended to cover), and
    the float probe below raised inside ShapeProp — silently costing every
    Opaque node its recorded shape, which downstream inference needs to survive
    a hole.
    """
    import torch.nn as nn

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        for user in node.users:
            if user.op == "call_module" and isinstance(
                modules.get(user.target), (nn.Embedding, nn.EmbeddingBag)
            ):
                return True
    return False


def _observed_shapes(gm, input_shape, fx_to_id, wants_int: bool = False) -> dict[str, list[int]]:
    """Per-node output shape via ShapeProp on the meta device — the same
    meta-device trick inference already relies on, so it costs no real memory.
    Fills Opaque nodes' recorded shape, which is what lets DOWNSTREAM inference
    survive a hole in the graph — so this is load-bearing for imported models,
    not the nicety the probe's dtype once assumed."""
    import copy

    import torch
    from torch.fx.passes.shape_prop import ShapeProp

    out: dict[str, list[int]] = {}
    try:
        # A COPY moved to meta: gm shares the caller's live submodules, and
        # nn.Module.to is in-place, so `gm.to("meta")` would strip the weights
        # off the very model we're about to read for state_dict transfer. The
        # copy is one-time and prototyping-scale.
        meta = copy.deepcopy(gm).to("meta")
        example = torch.zeros(
            *input_shape, device="meta",
            dtype=torch.long if wants_int else torch.float,
        )
        ShapeProp(meta).propagate(example)
        # Key by NAME, not identity: `meta` is a deepcopy, so every node in it
        # is a different object from the one `fx_to_id` was built against and
        # an identity lookup misses every time — which silently returned {} for
        # every model ever imported, leaving Opaque nodes with no recorded
        # shape for downstream inference to survive the hole with. fx node
        # names are stable across the copy.
        by_name = {node.name: nid for node, nid in fx_to_id.items()}
        for node in meta.graph.nodes:
            tm = node.meta.get("tensor_meta")
            nid = by_name.get(node.name)
            if nid is not None and tm is not None and hasattr(tm, "shape"):
                out[nid] = list(tm.shape)
    except Exception:
        pass  # observed shapes are a nicety; the graph stands without them
    return out


def _fill_opaque_shapes(nodes, findings, observed) -> None:
    opaque_ids = {f["id"] for f in findings if f["kind"] == "opaque"}
    for node in nodes:
        if node.id in opaque_ids and node.id in observed:
            dims = observed[node.id]
            node.params["out_shape"] = ", ".join(map(str, dims))


def layout(graph: Graph) -> None:
    """Assign positions by longest-path layering over the topo order, so an
    imported graph reads left-to-right instead of stacking at the origin.

    Deliberately simple (no dagre/elk dependency for this): each node's layer is
    1 + the max layer of its predecessors; within a layer, nodes stack by the
    barycenter of their predecessors' lanes. Good to ~80 nodes (resnet scale);
    long skip edges at densenet scale are a known limit, not a bug to solve with
    layout.
    """
    from .inference import build_incoming

    incoming = build_incoming(graph)
    preds: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for nid, ins in incoming.items():
        preds[nid] = [src for src, _ in ins.values()]

    # Longest-path layer, computed in topo order (fall back to input order if a
    # cycle somehow appears — it can't from fx, but stay total).
    order = [n.id for n in graph.nodes]
    layer: dict[str, int] = {}
    for nid in order:
        layer[nid] = 1 + max((layer.get(p, 0) for p in preds[nid]), default=-1)

    lanes: dict[int, list[str]] = {}
    for nid in order:
        lanes.setdefault(layer[nid], []).append(nid)

    X_STRIDE, Y_STRIDE = 260, 130
    by_id = {n.id: n for n in graph.nodes}
    for col, ids in sorted(lanes.items()):
        # Order within the column by the mean lane of predecessors, so edges
        # cross as little as this cheap heuristic manages.
        prev_lane = {nid: i for lane_ids in lanes.values() for i, nid in enumerate(lane_ids)}
        ids.sort(key=lambda nid: sum(prev_lane.get(p, 0) for p in preds[nid]) / max(1, len(preds[nid])))
        for row, nid in enumerate(ids):
            by_id[nid].position = NodePosition(x=col * X_STRIDE, y=row * Y_STRIDE)


def seed_from_weights(model, values: list, source_keys: list[str]) -> None:
    """Load an imported model's original weights into a freshly-generated module,
    positionally.

    This is the mechanism that makes "press Run on an import" real, and it works
    because the generated module names its members ``layer_0..layer_N`` in the
    same topo order fx walked — so the k-th generated state_dict entry is the
    k-th original one. Proven exact (maxdiff 0) on the resnet family.

    Guarded, not trusted: the key COUNT must match (a mismatch means a dropped
    or split layer — the tied-weight case), and each position's SHAPE must
    match. Either fails loud rather than mis-seeding, because silently loading
    the wrong tensor into the wrong layer is exactly the kind of quiet
    corruption this whole feature exists to prevent.
    """
    gen_state = model.state_dict()
    gen_keys = list(gen_state.keys())
    if len(gen_keys) != len(values):
        raise ImportError_(
            f"imported weights don't fit the generated model: {len(values)} source "
            f"tensors vs {len(gen_keys)} in the model. The graph was likely edited "
            f"after import, or a layer couldn't be represented — re-import to reseed."
        )
    if source_keys and len(source_keys) != len(gen_keys):
        raise ImportError_("imported state_dict key count changed since import — re-import")
    for i, (gk, val) in enumerate(zip(gen_keys, values)):
        if tuple(gen_state[gk].shape) != tuple(val.shape):
            sk = source_keys[i] if i < len(source_keys) else "?"
            raise ImportError_(
                f"imported weight #{i} ({sk} → {gk}) is {tuple(val.shape)} but the "
                f"generated layer expects {tuple(gen_state[gk].shape)} — refusing to "
                f"mis-seed. Re-import the model."
            )
    model.load_state_dict(dict(zip(gen_keys, values)))


# Re-export for callers that only need the param helpers.
__all__ = ["trace", "layout", "seed_from_weights", "ImportError_", "IGNORED_CTOR_ARGS"]
