"""Turn a graph into runnable PyTorch source — the heart of "runs exactly the
code it shows".

Emits the ``nn.Module`` class (one ``self.layer_N`` submodule per graph node, its
constructor args rendered from the registry) and the ``train()`` loop, plus the
dataloader helper. ``exec_generated`` is the single chokepoint that executes what
the UI displays, and ``layer_nodes`` maps generated ``layer_N`` names back to
canvas nodes for per-layer telemetry. No per-type branching beyond the structural
core (Input/Output/Concat/Add/Custom); every other node renders via its
registry ``emit``.
"""
import linecache
import re
from dataclasses import dataclass
from typing import NamedTuple

from .schema import Graph
from .inference import infer_shapes, build_incoming, resolve_custom, topo_order
from .registry import (
    DATA_PARAMS,
    REGISTRY,
    TRAINING_PARAMS,
    BackboneEmit,
    ModuleEmit,
    backbone_parts,
    causal_seq_axis,
    default_data,
    default_training,
    OpEmit,
    render_literal_args,
    render_module_args,
    render_op,
)

# The train/val split is carved with a fixed generator so the held-out set is
# identical across a run and every resume of it — the training seed still governs
# shuffling and weight init, but which samples validate is stable and comparable.
SPLIT_SEED = 1234

# The Positional Embedding node's spliced class — torch ships no built-in for
# learned position embeddings, so codegen emits this above the model exactly
# like a registered Custom class (self-contained exports/checkpoints).
_POSITIONAL_EMBEDDING_SOURCE = '''class PositionalEmbedding(nn.Module):
    """Learned position embeddings, added to a (batch, seq, embed) input."""

    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        return x + self.pos(torch.arange(x.size(1), device=x.device))
'''


def exec_generated(source: str, filename: str) -> dict:
    """Execute generated source in a fresh namespace and return that namespace —
    the single place Lamplighter runs the code it generates (the runner, the
    notebook ``build_*`` helpers, and checkpoint rebuilds all route here).

    The trust model, in one place: every source string that reaches this
    function was produced *in this process* by this module's templates (or a
    recipe's ``generate``) from a schema-validated design — interpolated values
    are escaped via ``repr()``, identifiers are validated (``_name_issues``,
    ``sanitize_class_name``) or checked against registry enums before they
    reach a template. It then runs in the user's own kernel with the user's
    own privileges — exactly like the notebook cells around it. Nothing that
    arrives over the network is executed without passing through codegen first.

    The source is registered with ``linecache`` under ``filename`` so a
    traceback raised *inside* generated code (a failing train(), a bad
    transform) shows the real source line instead of an opaque
    ``<lamplighter-…>`` marker. Use a distinct filename per source.
    """
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    ns: dict = {}
    exec(compile(source, filename, "exec"), ns)  # noqa: S102 — see docstring
    return ns


def sanitize_class_name(name: str) -> str:
    """A model's display name → a valid Python class identifier, e.g.
    ``"Generator"`` → ``Generator``, ``"Model 2"`` → ``Model2``, ``"my-gan"`` →
    ``MyGan``. Falls back to ``Model`` when nothing usable remains."""
    parts = re.findall(r"[A-Za-z0-9]+", str(name or ""))
    ident = "".join(p[:1].upper() + p[1:] for p in parts)
    if not ident or not ident[0].isalpha():
        ident = "Model" + ident
    return ident


def class_name_for(name: str, sole: bool) -> str:
    """The generated class name for a model. A lone model keeps the classic
    ``GeneratedModel`` (single-model output stays byte-identical); when several
    models coexist each takes its own sanitized name so they read as
    ``class Generator`` / ``class Discriminator``."""
    return "GeneratedModel" if sole else sanitize_class_name(name)


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


@dataclass(frozen=True)
class _MetricSpec:
    """One optional epoch metric, as generated-code templates. ``{p}`` is the
    accumulator prefix ("" in the train loop, "v" in val — the correct/vcorrect
    convention), ``{seen}`` the sample counter, ``{result}`` the epoch variable
    (train_acc / val_acc). ``losses`` gates the metric to losses where it's
    meaningful (accuracy's silent-gate precedent: a regression loss simply
    doesn't emit classification-metric code)."""

    key: str  # history suffix: train_<key> / val_<key>
    fmt: str  # printed precision
    losses: tuple[str, ...]
    init: tuple[str, ...]  # per-epoch accumulator setup
    update: tuple[str, ...]  # per-batch, after bs/seen exist (out/yb in scope)
    finalize: tuple[str, ...]  # epoch end; the last line assigns {result}


_CLS = ("CrossEntropyLoss", "NLLLoss")
# Losses that accept a class-imbalance argument (CE/NLL take a per-class
# `weight` vector; BCEWithLogits takes a `pos_weight` scale for the positive
# class). Shared with the training form's show_if so the toggle and the code
# agree on where it applies.
_WEIGHTABLE_LOSSES = ("CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss")
_METRIC_SPECS: dict[str, _MetricSpec] = {
    "accuracy": _MetricSpec(
        key="acc", fmt=".3f", losses=_CLS,
        init=("{p}correct = 0",),
        update=("{p}correct += (out.argmax(dim=-1) == yb).sum().item()",),
        finalize=("{result} = {p}correct / {seen}",),
    ),
    "top5_accuracy": _MetricSpec(
        key="top5", fmt=".3f", losses=_CLS,
        init=("{p}top5 = 0",),
        # min() so a head with < 5 classes still runs (top-k of all classes).
        update=(
            "{p}top5 += (out.topk(min(5, out.size(-1)), dim=-1).indices == yb.unsqueeze(-1)).any(dim=-1).sum().item()",
        ),
        finalize=("{result} = {p}top5 / {seen}",),
    ),
    "macro_f1": _MetricSpec(
        key="f1", fmt=".3f", losses=_CLS,
        init=("{p}f1_preds, {p}f1_targs = [], []",),
        update=(
            "{p}f1_preds.append(out.argmax(dim=-1).cpu())",
            "{p}f1_targs.append(yb.cpu())",
        ),
        # Per-class TP/FP/FN over the whole epoch, F1 averaged across classes.
        finalize=(
            "{p}f1_pred = torch.cat({p}f1_preds)",
            "{p}f1_targ = torch.cat({p}f1_targs)",
            "{p}f1_C = max(int({p}f1_pred.max()), int({p}f1_targ.max())) + 1",
            "{p}f1_tp = torch.bincount({p}f1_targ[{p}f1_pred == {p}f1_targ], minlength={p}f1_C).float()",
            "{p}f1_fp = torch.bincount({p}f1_pred, minlength={p}f1_C).float() - {p}f1_tp",
            "{p}f1_fn = torch.bincount({p}f1_targ, minlength={p}f1_C).float() - {p}f1_tp",
            "{result} = (2 * {p}f1_tp / (2 * {p}f1_tp + {p}f1_fp + {p}f1_fn).clamp(min=1e-12)).mean().item()",
        ),
    ),
    "mae": _MetricSpec(
        key="mae", fmt=".4f", losses=("MSELoss", "L1Loss", "HuberLoss"),
        init=("{p}abs_err = 0.0",),
        update=("{p}abs_err += (out - yb).abs().mean().item() * bs",),
        finalize=("{result} = {p}abs_err / {seen}",),
    ),
}


# Spliced above train() when class weighting is on. Reading a dataset's own
# labels matters: counting by iterating an ImageFolder would decode every
# image just to read its class.
_LABEL_COUNTS_SOURCE = '''def label_counts(loader, n_classes):
    """How many training samples per class — the basis for inverse-frequency
    weights. Reads the dataset's labels directly when it exposes them
    (`.targets` on torchvision datasets and ImageFolder, `.tensors` on a
    TensorDataset, either behind a random_split Subset); otherwise makes one
    pass over the batches."""
    dataset = loader.dataset
    indices = getattr(dataset, "indices", None)  # a random_split Subset
    if indices is not None:
        dataset = dataset.dataset
    targets = getattr(dataset, "targets", None)
    if targets is None and hasattr(dataset, "tensors"):
        targets = dataset.tensors[-1]
    if targets is None:
        counts = torch.zeros(n_classes)
        for batch in loader:
            counts += torch.bincount(batch[-1].flatten().long().cpu(), minlength=n_classes).float()
        return counts
    targets = torch.as_tensor(targets).flatten().long()
    if indices is not None:
        targets = targets[list(indices)]
    return torch.bincount(targets, minlength=n_classes).float()
'''

# The in-train() weight computation, per loss family.
_CLASS_WEIGHT_LINES = [
    "    # Inverse-frequency class weights over the training split: a rarer",
    "    # class costs more when missed. The width comes from the model's own",
    "    # logits (one 1-sample probe under eval, so BatchNorm's running stats",
    "    # stay untouched), so the vector lines up even if the split is missing",
    "    # a class entirely.",
    "    was_training = model.training",
    "    model.eval()",
    "    with torch.no_grad():",
    "        probe = next(iter(loader))",
    "        n_classes = model(*[t[:1].to(device) for t in probe[:-1]]).size(-1)",
    "    model.train(was_training)",
    "    counts = label_counts(loader, n_classes)",
    "    weight = (counts.sum() / (n_classes * counts.clamp(min=1.0))).to(device)",
]

_POS_WEIGHT_LINES = [
    "    # How much rarer the positive class is — BCEWithLogits scales the",
    "    # positive term rather than taking a per-class vector. A multi-label",
    "    # target is pooled across its columns into one global ratio.",
    "    counts = label_counts(loader, 2)",
    "    pos_weight = (counts[0] / counts[1].clamp(min=1.0)).to(device)",
]


def _history_keys(metric_key: str | None, include_val: bool) -> list[str]:
    """Ordered metric keys for the returned per-epoch history dict."""
    keys = ["train_loss"]
    if metric_key:
        keys.append(f"train_{metric_key}")
    if include_val:
        keys.append("val_loss")
        if metric_key:
            keys.append(f"val_{metric_key}")
    return keys


def _history_init_line(keys: list[str]) -> str:
    """`    history = {"train_loss": [], …}` — one empty list per tracked metric."""
    return "    history = {" + ", ".join(f'"{k}": []' for k in keys) + "}"


def device_resolve_lines() -> list[str]:
    """Generated preamble that turns the `device` arg into a torch.device. "auto"
    prefers CUDA, then MPS (guarded for torch builds without the mps backend),
    else CPU; a specific name is used as-is. Shared by every recipe's train() so
    the accelerator logic lives once, not once per loop template."""
    return [
        '    if device == "auto":',
        "        if torch.cuda.is_available():",
        '            device = "cuda"',
        '        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():',
        '            device = "mps"',
        "        else:",
        '            device = "cpu"',
        "    device = torch.device(device)",
    ]


def _device_resolution_lines() -> list[str]:
    """The supervised loop's preamble: resolve the device, then move the model
    onto it (recipes move their own role modules, so that line is theirs)."""
    return [*device_resolve_lines(), "    model = model.to(device)"]


def _module_nodes(node_map: dict, order: list[str], live: set) -> list[str]:
    """Node ids that become ``self.layer_N`` members, in codegen (midx) order —
    a Custom node, a Backbone, or any ModuleEmit node, walked in topo order and
    restricted to the live subgraph. The single source of truth for both generate_module's
    naming and layer_nodes' mapping, so the two can't drift. (Input/Output/
    Concat/Add carry no module; OpEmit nodes render inline, so all are skipped.)"""
    result: list[str] = []
    for nid in order:
        if nid not in live:
            continue
        t = node_map[nid].type
        if t in ("Custom", "PositionalEmbedding"):  # spliced-class modules
            result.append(nid)
            continue
        node_def = REGISTRY.get(t)
        if isinstance(getattr(node_def, "emit", None), (ModuleEmit, BackboneEmit)):
            result.append(nid)
    return result


class LayerNode(NamedTuple):
    """One generated ``self.layer_N`` module, mapped back to its canvas node."""

    layer: str  # the generated attribute name, e.g. "layer_0"
    node_id: str  # the canvas node id (for badges / drill-in)
    label: str  # the node's user name if set, else its type
    type: str  # the node type (Conv2d, Linear, Custom, …)


def layer_nodes(graph: Graph) -> list[LayerNode]:
    """Map each generated ``self.layer_N`` module back to its canvas node, in the
    order generate_module names them. Lets the runner label per-layer stats (e.g.
    training-health readouts) by node instead of an opaque index. Shares
    _module_nodes with generate_module, so the mapping never drifts from the
    actual attribute names."""
    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    order, _ = topo_order(graph, incoming)
    live = _live_nodes(graph, incoming, node_map)
    return [
        LayerNode(f"layer_{i}", nid, _node_name(node_map[nid]) or node_map[nid].type, node_map[nid].type)
        for i, nid in enumerate(_module_nodes(node_map, order, live))
    ]


def generate_module(graph: Graph, class_name: str = "GeneratedModel") -> str:
    """The graph's ``nn.Module`` source. ``class_name`` names the generated class
    — the default keeps single-model output byte-identical; a project gives each
    model its own sanitized name (``Generator``, ``Discriminator``) so several
    modules can coexist in one namespace."""
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
    # layer_N indices come from the shared _module_nodes order, so the attribute
    # names here stay in lockstep with layer_nodes()'s node mapping.
    midx_of = {nid: i for i, nid in enumerate(_module_nodes(node_map, order, live))}

    init_lines: list[str] = []
    fwd_lines: list[str] = []
    custom_sources: dict[str, str] = {}  # spliced class name → its source
    weight_enums: set[str] = set()  # torchvision weights enums to import
    frozen_attrs: list[str] = []  # frozen backbones — see the train() override

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

        if t == "Add":
            handles = sorted(incoming[nid])
            args = [var[incoming[nid][h]] for h in handles]
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = {' + '.join(args)}")
            continue

        if t == "PositionalEmbedding":
            # Learned position embeddings — no nn built-in exists, so a fixed
            # generated class is spliced above the model (the Custom-node
            # mechanism) and instantiated like any layer. The embedding dim
            # follows the input's last dim; max_len is the node's knob.
            if class_name == "PositionalEmbedding":
                raise ValueError("the model's class name clashes with the PositionalEmbedding node — rename the model")
            if custom_sources.get("PositionalEmbedding", _POSITIONAL_EMBEDDING_SOURCE) != _POSITIONAL_EMBEDDING_SOURCE:
                raise ValueError(
                    "a registered custom module already uses the class name 'PositionalEmbedding'"
                )
            custom_sources["PositionalEmbedding"] = _POSITIONAL_EMBEDDING_SOURCE
            input_shape = shapes[incoming[nid]["input"]]
            max_len = int(p.get("max_len", 512))
            init_lines.append(
                f"self.layer_{midx_of[nid]} = PositionalEmbedding({max_len}, {input_shape[-1]})"
            )
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = self.layer_{midx_of[nid]}({sv(nid)})")
            continue

        if t == "Custom":
            # A registered notebook class: splice its source into this module
            # (so exports/checkpoints stay self-contained) and instantiate it
            # with the node's literal args.
            import inspect
            import textwrap

            cls, pos_args, kw_args = resolve_custom(p)
            cname = cls.__name__
            if cname == class_name:
                raise ValueError(f"the custom module '{cname}' clashes with the model's class name")
            try:
                source = textwrap.dedent(inspect.getsource(cls))
            except (OSError, TypeError):
                raise ValueError(
                    f"cannot read the source of {cname} — define it in a notebook "
                    "cell (dynamically-built classes aren't supported)"
                ) from None
            if custom_sources.get(cname, source) != source:
                raise ValueError(f"two registered modules share the class name '{cname}'")
            custom_sources[cname] = source
            rendered = render_literal_args(pos_args, kw_args)
            init_lines.append(f"self.layer_{midx_of[nid]} = {cname}({rendered})")
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = self.layer_{midx_of[nid]}({sv(nid)})")
            continue

        # Standard nodes render an nn.<cls> member + call, built from the same
        # args inference uses (so code and shapes can't disagree).
        node_def = REGISTRY.get(t)
        emit = node_def.emit if node_def else None

        if isinstance(emit, OpEmit):
            # A functional op: the same rendered expression inference eval'd.
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = {render_op(node_def, p, sv(nid))}")
            continue

        if isinstance(emit, BackboneEmit):
            # Build it, strip the classifier head (this node yields FEATURES —
            # the head is drawn on the canvas), and optionally freeze. All three
            # steps are emitted, so the shown code is the whole story.
            spec, pretrained, freeze = backbone_parts(node_def, p)
            attr = f"self.layer_{midx_of[nid]}"
            weights = f"{spec.weights}.DEFAULT" if pretrained else "None"
            if pretrained:
                weight_enums.add(spec.weights)
            init_lines.append(f"{attr} = models.{spec.ctor}(weights={weights})")
            init_lines.append(f"{attr}.{spec.head} = nn.Identity()  # features, not logits")
            if freeze:
                init_lines.append(f"for p in {attr}.parameters():")
                init_lines.append("    p.requires_grad = False")
                frozen_attrs.append(attr)
            v = f"t{counter}"
            counter += 1
            var[(nid, "output")] = v
            fwd_lines.append(f"{v} = {attr}({sv(nid)})")
            continue

        if isinstance(emit, ModuleEmit):
            input_shape = shapes[incoming[nid]["input"]]
            rendered = render_module_args(node_def, p, input_shape)
            init_lines.append(f"self.layer_{midx_of[nid]} = nn.{emit.cls}({rendered})")
            result = f"t{counter}"
            counter += 1
            # call_repeat > 1: the input repeats as every argument (self-
            # attention renders as `self.layer_N(x, x, x)`).
            call_args = ", ".join([sv(nid)] * emit.call_repeat)
            # Causal masking is a call-time concern: the mask is sized to the
            # sequence the batch actually carries, so it's built in forward().
            axis = causal_seq_axis(node_def, p)
            if axis is not None:
                x = sv(nid)
                call_args += (
                    f", {emit.causal_mask_kwarg}=nn.Transformer."
                    f"generate_square_subsequent_mask({x}.size({axis}), device={x}.device)"
                    ", is_causal=True"
                )
            fwd_lines.append(f"{result} = self.layer_{midx_of[nid]}({call_args})")
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
    if any(isinstance(getattr(REGISTRY.get(node_map[nid].type), "emit", None), BackboneEmit)
           for nid in live):
        header.append("from torchvision import models")
        if weight_enums:
            header.append(f"from torchvision.models import {', '.join(sorted(weight_enums))}")
    if named:
        field_list = ", ".join(repr(f) for f in fields)
        header += [
            "from collections import namedtuple",
            "",
            "",
            f'ModelOutput = namedtuple("ModelOutput", [{field_list}])',
        ]

    parts = list(header)
    # Registered custom classes, spliced verbatim above the model so the
    # generated module is self-contained (exports and checkpoints rebuild
    # without the notebook session).
    for source in custom_sources.values():
        parts += ["", "", source.rstrip("\n")]
    parts += [
        "",
        "",
        f"class {class_name}(nn.Module):",
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

    if frozen_attrs:
        # requires_grad=False stops the WEIGHTS moving, but BatchNorm's running
        # statistics keep updating in train mode — so a "frozen" backbone would
        # still drift between epochs, silently. Pinning it to eval mode is what
        # freezing has to mean.
        parts += [
            "",
            "    def train(self, mode=True):",
            '        """Keep frozen backbones in eval mode: their BatchNorm layers would',
            '        otherwise keep updating running statistics, so a frozen model would',
            '        still drift between epochs."""',
            "        super().train(mode)",
            *[f"        {attr}.eval()" for attr in frozen_attrs],
            "        return self",
        ]

    return "\n".join(parts) + "\n"


def generate_eval(graph: Graph, training: dict) -> str:
    """An ``evaluate(model, loader)`` — a run's verdict on data it never trained
    on. Deliberately the PLAIN objective: no label smoothing (a regularizer) and
    no class weighting (a rebalancer), because a test number has to mean the
    same thing across runs that trained with different remedies. Reports the
    loss, the configured metric (gated on the loss, as everywhere), and the
    sample count it saw — a score without its n isn't a result."""
    cfg = {**default_training(), **(training or {})}
    loss = str(cfg["loss"])
    device = str(cfg["device"])
    allowed = next(p.choices for p in TRAINING_PARAMS if p.name == "loss")
    if loss not in allowed:
        raise ValueError(f"unknown loss '{loss}' — expected one of: {', '.join(allowed)}")

    spec = _METRIC_SPECS.get(str(cfg["metric"]))
    if spec is not None and loss not in spec.losses:
        spec = None
    loss_call, loss_source = _loss_expression(cfg, loss, weighted=False, smoothing=False)

    incoming = build_incoming(graph)
    node_map = {n.id: n for n in graph.nodes}
    multi = len(model_inputs(graph, incoming, node_map)) > 1
    unpack, to_dev, call = (
        ("*xb, yb = batch", "xb = [t.to(device) for t in xb]", "model(*xb)")
        if multi
        else ("xb, yb = batch", "xb = xb.to(device)", "model(xb)")
    )

    lines = ["import torch", "import torch.nn as nn", "", ""]
    if loss_source:
        lines += [loss_source.rstrip("\n"), "", ""]
    lines += [
        f"def evaluate(model, loader, *, device={device!r}):",
        *_device_resolution_lines(),
        f"    loss_fn = {loss_call}",
        "    model.eval()",
        "    total, seen = 0.0, 0",
        *(["    " + t.format(p="") for t in spec.init] if spec else []),
        "    with torch.no_grad():",
        "        for batch in loader:",
        f"            {unpack}",
        f"            {to_dev}",
        "            yb = yb.to(device)",
        f"            out = {call}",
        "            bs = yb.size(0)",
        "            total += loss_fn(out, yb).item() * bs",
        "            seen += bs",
        *(["            " + t.format(p="") for t in spec.update] if spec else []),
        '    result = {"test_loss": total / seen, "n": seen}',
    ]
    if spec:
        lines += ["    " + t.format(p="", seen="seen", result=f"test_{spec.key}") for t in spec.finalize]
        lines.append(f'    result["test_{spec.key}"] = test_{spec.key}')
    lines.append("    return result")
    return "\n".join(lines) + "\n"


def _has_frozen_params(graph: Graph) -> bool:
    """Does this graph freeze any weights? Read from the registry rather than by
    node type, so any future emit kind that can freeze is covered by declaring
    its param — the same rule that keeps the engines free of per-type branches."""
    node_map = {n.id: n for n in graph.nodes}
    incoming = build_incoming(graph)
    for nid in _live_nodes(graph, incoming, node_map):
        node = node_map[nid]
        emit = getattr(REGISTRY.get(node.type), "emit", None)
        if isinstance(emit, BackboneEmit) and bool(node.params.get(emit.freeze_param, True)):
            return True
    return False


def generate_sampling(block_size: int, device: str = "cpu") -> str:
    """A ``generate(model, prompt_ids)`` that samples a continuation one token
    at a time — the only way to see what a language model actually learned (a
    loss curve says how surprised it is, not what it writes). Standard
    autoregressive decoding: take the last position's logits, divide by the
    temperature, sample. The window slides so the model never sees more context
    than it was trained on."""
    return "\n".join([
        "import torch",
        "",
        "",
        "@torch.no_grad()",
        f"def generate(model, prompt_ids, *, max_new_tokens=200, temperature=1.0, "
        f"block_size={int(block_size)}, device={device!r}):",
        "    device = torch.device(device)",
        "    model = model.to(device).eval()",
        "    ids = prompt_ids.flatten().long().to(device)",
        "    for _ in range(max_new_tokens):",
        "        # Only the most recent block_size tokens — the context the",
        "        # model was trained to read.",
        "        window = ids[-block_size:].unsqueeze(0)",
        "        logits = model(window)[0, -1]",
        "        # Temperature flattens (>1) or sharpens (<1) the distribution;",
        "        # near zero it becomes a greedy argmax.",
        "        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)",
        "        ids = torch.cat([ids, torch.multinomial(probs, num_samples=1)])",
        "    return ids",
    ]) + "\n"


def _loss_expression(cfg: dict, loss: str, *, weighted: bool, smoothing: bool) -> tuple[str, str]:
    """(the ``loss_fn = …`` expression, any class source to splice above it).

    A Custom loss resolves a registered nn.Module class and is spliced verbatim
    — the Custom node's machinery, so exports and checkpoints stay
    self-contained. ``weighted``/``smoothing`` are how evaluate() asks for the
    PLAIN objective: class weighting and label smoothing are training-time
    devices (one rebalances, the other regularizes), and a test number has to
    mean the same thing across runs that used them differently."""
    if loss == "Custom":
        import inspect
        import textwrap

        cls, pos_args, kw_args = resolve_custom(
            {"cls": cfg.get("loss_cls"), "args": cfg.get("loss_args")}
        )
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            raise ValueError(
                f"cannot read the source of {cls.__name__} — define it in a notebook "
                "cell (dynamically-built classes aren't supported)"
            ) from None
        return f"{cls.__name__}({render_literal_args(pos_args, kw_args)})", source

    loss_kwargs = []
    smooth = float(cfg.get("label_smoothing", 0.0) or 0.0)
    if smoothing and loss == "CrossEntropyLoss" and smooth:
        loss_kwargs.append(f"label_smoothing={smooth!r}")
    if weighted:
        loss_kwargs.append("pos_weight=pos_weight" if loss == "BCEWithLogitsLoss" else "weight=weight")
    return f"nn.{loss}({', '.join(loss_kwargs)})", ""


def generate_training(graph: Graph, training: dict) -> str:
    """A self-contained ``train(model, loader)`` from the ``training`` config
    (loss/optimizer/hyperparams, metric, device) — a project-level concern, passed
    in rather than read off the graph. ``graph`` supplies only the model's shape
    (its input count). Data always arrives as a torch DataLoader built by the Data
    panel's ``make_dataloaders()`` — one data path, so what runs is exactly what
    both panels show. An optional ``val_loader`` runs validation; ``on_epoch``
    reports per-epoch metrics and supports early stopping (return False to stop)."""
    cfg = {**default_training(), **(training or {})}
    loss = str(cfg["loss"])
    optimizer = str(cfg["optimizer"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    metric = str(cfg["metric"])
    device = str(cfg["device"])

    # Loss and optimizer names land in the source as attributes (nn.X /
    # torch.optim.X), so like the scheduler and the torchvision dataset they
    # MUST be validated, not escaped — a raw API caller can't inject code.
    allowed = next(p.choices for p in TRAINING_PARAMS if p.name == "loss")
    if loss not in allowed:
        raise ValueError(f"unknown loss '{loss}' — expected one of: {', '.join(allowed)}")
    allowed = next(p.choices for p in TRAINING_PARAMS if p.name == "optimizer")
    if optimizer not in allowed:
        raise ValueError(f"unknown optimizer '{optimizer}' — expected one of: {', '.join(allowed)}")

    # Class weighting: rebalance what a mistake COSTS, by inverse class
    # frequency over the training split. Gated to the losses that take it —
    # a weight vector for CE/NLL, a positive-class scale for BCEWithLogits
    # (different arguments, different arithmetic) — so an off-target config
    # emits nothing, like the metric specs.
    weighted = bool(cfg.get("class_weights", False)) and loss in _WEIGHTABLE_LOSSES

    loss_call, loss_source = _loss_expression(cfg, loss, weighted=weighted, smoothing=True)

    # Gradient accumulation: step every N batches with the loss scaled by 1/N.
    # 1 keeps the plain per-batch loop, byte-identical to an unset form.
    accum = max(1, int(cfg.get("accumulate_steps") or 1))

    # Each metric gates on the losses it's meaningful for (accuracy's
    # precedent) — a regression loss never emits classification-metric code,
    # and vice versa. Unknown/"none" emit nothing.
    spec = _METRIC_SPECS.get(metric)
    if spec is not None and loss not in spec.losses:
        spec = None

    # A multi-input model's loader yields (x0, x1, …, y): `*xb, yb = batch`
    # unpacks the trailing target, the rest feed model(*xb).
    incoming = build_incoming(graph)
    node_map = {n.id: n for n in graph.nodes}
    multi = len(model_inputs(graph, incoming, node_map)) > 1

    opt_args = [f"lr={lr!r}"]
    # Momentum applies only to the optimizers that take it (the Adam family's
    # momentum lives in betas); 0 emits nothing — torch's own default.
    momentum = float(cfg.get("momentum", 0.9) or 0.0)
    if optimizer in ("SGD", "RMSprop") and momentum != 0.0:
        opt_args.append(f"momentum={momentum!r}")
    if weight_decay != 0.0:  # omit the default for cleaner code
        opt_args.append(f"weight_decay={weight_decay!r}")
    # With a frozen backbone in the model, the optimizer takes only what can
    # actually move. (Passing frozen params is harmless for SGD/Adam but wrong
    # for weight decay, which would keep shrinking them.) Gated on the graph, so
    # a model with nothing frozen generates the plain line it always did.
    params_expr = (
        "[p for p in model.parameters() if p.requires_grad]"
        if _has_frozen_params(graph)
        else "model.parameters()"
    )
    opt_call = f"torch.optim.{optimizer}({params_expr}, {', '.join(opt_args)})"

    # Optional LR schedule. "none" (the default) emits nothing, so unscheduled
    # runs generate byte-identical source. The scheduler name is an enum but is
    # re-checked here (same rule as the torchvision dataset name: it lands in
    # the source as an attribute, so it must be validated, not escaped).
    scheduler = str(cfg.get("scheduler", "none"))
    per_step_sched = scheduler == "OneCycleLR"  # steps per BATCH, not per epoch
    if scheduler == "StepLR":
        sched_call = (
            f"torch.optim.lr_scheduler.StepLR(opt, step_size={int(cfg['step_size'])}, "
            f"gamma={float(cfg['gamma'])!r})"
        )
        sched_step = "sched.step()"
    elif scheduler == "CosineAnnealingLR":
        # T_max = the epochs arg, so the anneal spans exactly this run.
        sched_call = "torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)"
        sched_step = "sched.step()"
    elif scheduler == "OneCycleLR":
        # Warmup + anneal in one standard package. The form lr is the PEAK
        # (max_lr), and the schedule is sized to the whole run and stepped per
        # batch — the sched.step() lands inside the batch loop below.
        # Sized in optimizer steps — with accumulation that's ceil(batches/N).
        steps_expr = f"(len(loader) + {accum - 1}) // {accum}" if accum > 1 else "len(loader)"
        sched_call = (
            f"torch.optim.lr_scheduler.OneCycleLR(opt, max_lr={lr!r}, "
            f"epochs=epochs, steps_per_epoch={steps_expr})"
        )
        sched_step = None
    elif scheduler == "ReduceLROnPlateau":
        sched_call = (
            f"torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "
            f"factor={float(cfg['plateau_factor'])!r}, patience={int(cfg['plateau_patience'])})"
        )
        # Plateau steps on the metric: val loss when validation runs, else train.
        sched_step = 'sched.step(history["val_loss"][-1] if val_loader is not None else train_loss)'
    elif scheduler != "none":
        raise ValueError(f"unknown LR scheduler '{scheduler}'")

    # Gradient clipping and mixed precision, both off by default (emitting
    # nothing, so the plain loop stays byte-identical). Under AMP the gradients
    # must be unscaled before clipping — the generated order encodes that.
    raw_clip = cfg.get("clip_grad_norm")
    clip = float(raw_clip) if raw_clip is not None else None
    amp = bool(cfg.get("amp", False))

    if multi:
        unpack, to_dev, call = "*xb, yb = batch", "xb = [t.to(device) for t in xb]", "model(*xb)"
    else:
        unpack, to_dev, call = "xb, yb = batch", "xb = xb.to(device)", "model(xb)"

    lines = ["import torch", "import torch.nn as nn", "", ""]
    if loss_source:
        # The custom loss class, spliced verbatim (the Custom node's rule) so
        # the source runs standalone and checkpoints/ejects stay self-contained.
        lines += [loss_source.rstrip("\n"), "", ""]
    if weighted:
        lines += [_LABEL_COUNTS_SOURCE.rstrip("\n"), "", ""]
    lines.append(
        f"def train(model, loader, *, epochs={epochs}, val_loader=None, device={device!r}, on_epoch=None, on_step=None):"
    )
    lines += _device_resolution_lines()
    if weighted:
        lines += _POS_WEIGHT_LINES if loss == "BCEWithLogitsLoss" else _CLASS_WEIGHT_LINES
    # Val keys are always present (val_loader may be passed at call time); their
    # lists stay empty when no val_loader is given.
    history_keys = _history_keys(spec.key if spec else None, include_val=True)
    lines += [
        f"    loss_fn = {loss_call}",
        f"    opt = {opt_call}",
    ]
    if amp:
        lines.append("    scaler = torch.amp.GradScaler(device.type)")
    if scheduler != "none":
        lines.append(f"    sched = {sched_call}")
        history_keys = history_keys + ["lr"]
    lines += [
        _history_init_line(history_keys),
        "    step = 0",
        "    for epoch in range(epochs):",
        "        model.train()",
        "        running, seen = 0.0, 0",
    ]
    if spec:
        lines += ["        " + t.format(p="") for t in spec.init]
    # Forward/backward and the optimizer-step ops, assembled separately so
    # accumulation can reuse the step ops at the boundary AND the tail flush.
    # With accumulation the loss is scaled by 1/N (gradients average); the
    # reported batch_loss stays the UNscaled value.
    scaled = f"(loss / {accum})" if accum > 1 else "loss"
    if amp:
        fwd_lines = [
            "with torch.autocast(device_type=device.type):",
            f"    out = {call}",
            "    loss = loss_fn(out, yb)",
            f"scaler.scale({scaled}).backward()",
        ]
        step_ops = []
        if clip is not None:
            step_ops += [
                "scaler.unscale_(opt)",  # clip real gradients, not scaled ones
                f"torch.nn.utils.clip_grad_norm_(model.parameters(), {clip!r})",
            ]
        step_ops += ["scaler.step(opt)", "scaler.update()"]
    else:
        fwd_lines = [f"out = {call}", "loss = loss_fn(out, yb)", f"{scaled}.backward()"]
        step_ops = []
        if clip is not None:
            step_ops.append(f"torch.nn.utils.clip_grad_norm_(model.parameters(), {clip!r})")
        step_ops.append("opt.step()")
    if per_step_sched:
        step_ops.append("sched.step()")  # OneCycle advances every optimizer step

    report_lines = [
        "            bs = yb.size(0)",
        "            batch_loss = loss.item()",
        "            running += batch_loss * bs",
        "            seen += bs",
        "            step += 1",
        "            if on_step is not None:",
        '                on_step(step, {"train_loss": batch_loss})',
    ]
    if accum > 1:
        # Gradients build across `accum` batches, the optimizer steps at each
        # boundary, and a ragged tail still flushes — an effective batch of
        # accum × batch_size without the memory. `micro` counts batches since
        # the last optimizer step; initialized here so an empty loader (which
        # skips the body entirely) still reads as "nothing to flush".
        lines += [
            "        micro = 0",
            "        opt.zero_grad()",
            "        for batch in loader:",
            f"            {unpack}",
            f"            {to_dev}",
            "            yb = yb.to(device)",
            *["            " + line for line in fwd_lines],
            "            micro += 1",
            f"            if micro % {accum} == 0:",
            *["                " + line for line in step_ops],
            "                opt.zero_grad()",
            *report_lines,
        ]
    else:
        lines += [
            "        for batch in loader:",
            f"            {unpack}",
            f"            {to_dev}",
            "            yb = yb.to(device)",
            "            opt.zero_grad()",
            *["            " + line for line in fwd_lines + step_ops],
            *report_lines,
        ]
    if spec:
        lines += ["            " + t.format(p="") for t in spec.update]
    if accum > 1:
        lines += [
            f"        if micro % {accum}:  # flush the ragged tail",
            *["            " + line for line in step_ops],
            "            opt.zero_grad()",
        ]
    lines.append("        train_loss = running / seen")
    if spec:
        lines += ["        " + t.format(p="", seen="seen", result=f"train_{spec.key}") for t in spec.finalize]
    # The report is built at run time because val is optional (val_loader=None).
    lines.append('        msg = f"epoch {epoch + 1}/{epochs}  loss {train_loss:.4f}"')
    if spec:
        lines.append(f'        msg += f" {spec.key} {{train_{spec.key}:{spec.fmt}}}"')

    lines += [
        "        if val_loader is not None:",
        "            model.eval()",
        "            vloss, vseen = 0.0, 0",
    ]
    if spec:
        lines += ["            " + t.format(p="v") for t in spec.init]
    val_fwd = (
        ["                    with torch.autocast(device_type=device.type):",
         f"                        out = {call}"]
        if amp
        else [f"                    out = {call}"]
    )
    lines += [
        "            with torch.no_grad():",
        "                for batch in val_loader:",
        f"                    {unpack}",
        f"                    {to_dev}",
        "                    yb = yb.to(device)",
        *val_fwd,
        "                    bs = yb.size(0)",
        "                    vloss += loss_fn(out, yb).item() * bs",
        "                    vseen += bs",
    ]
    if spec:
        lines += ["                    " + t.format(p="v") for t in spec.update]
    lines.append("            val_loss = vloss / vseen")
    if spec:
        lines += ["            " + t.format(p="v", seen="vseen", result=f"val_{spec.key}") for t in spec.finalize]
    lines.append('            msg += f"  val_loss {val_loss:.4f}"')
    if spec:
        lines.append(f'            msg += f" val_{spec.key} {{val_{spec.key}:{spec.fmt}}}"')
    # Val metrics recorded only on epochs where a val_loader ran.
    lines.append('            history["val_loss"].append(val_loss)')
    if spec:
        lines.append(f'            history["val_{spec.key}"].append(val_{spec.key})')

    # Print only when running standalone: an on_epoch consumer (the app's in-kernel
    # runner) reports progress itself, so printing there just leaks into the notebook.
    lines.append("        if on_epoch is None:")
    lines.append("            print(msg)")
    lines.append('        history["train_loss"].append(train_loss)')
    if spec:
        lines.append(f'        history["train_{spec.key}"].append(train_{spec.key})')
    if scheduler != "none":
        # Record the lr this epoch trained at, THEN advance the schedule (a
        # per-step schedule already advanced inside the loop — record only).
        lines.append('        history["lr"].append(opt.param_groups[0]["lr"])')
        if sched_step is not None:
            lines.append(f"        {sched_step}")
    # Per-epoch hook: progress reporting and early stopping (return False to stop).
    lines.append("        if on_epoch is not None and on_epoch(epoch + 1, history) is False:")
    lines.append("            break")
    lines.append("    return history")
    return "\n".join(lines) + "\n"


def generate_dataloader(
    graph: Graph, data: dict, namespace: dict | None = None,
    needs_targets: bool = True, has_val: bool = True,
) -> str:
    """A `make_dataloaders()` helper from the ``data`` config (source, batching —
    passed in, not read off the graph), returning (train_loader, val_loader). It
    pairs with the generated train():
    `train_loader, val_loader = make_dataloaders(...)` then
    `train(model, train_loader, val_loader=val_loader)`. `namespace` (defaults to
    the session's data registry; injectable for tests) lets the memory source
    specialize by the picked object's type. `needs_targets=False` (an adversarial
    recipe) builds an unlabeled loader over X alone — batches of `(x,)`;
    `has_val=False` (a recipe whose loop never validates) zeroes the held-out
    split, so a stale val_split can't silently carve off training data."""
    if namespace is None:
        from .datastore import registry

        namespace = registry()
    cfg = {**default_data(), **(data or {})}
    source = str(cfg["source"])
    batch_size = int(cfg["batch_size"])
    shuffle = bool(cfg["shuffle"])
    # Only the train loader drops a ragged batch; omitted when off for clean code.
    drop = ", drop_last=True" if bool(cfg["drop_last"]) else ""
    common = _loader_common(cfg)  # num_workers / pin_memory, on every loader
    # A multi-input model needs make_dataloaders(X0, X1, y) → TensorDataset(X0, X1, y).
    # LIVE inputs only — the same liveness rule module/training codegen use, so a
    # stray Input node on the canvas can't skew the signature the runner calls.
    node_map = {n.id: n for n in graph.nodes}
    n_inputs = len(model_inputs(graph, build_incoming(graph), node_map)) or 1
    if source == "sequence":
        # Type-aware like the memory source: a picked STRING gets a character
        # tokenizer built from it, a picked tensor is already token ids.
        from .introspect import variable_kind

        picked = str(cfg.get("tokens_var", "") or "").strip()
        vocab = (
            char_vocab(namespace[picked])
            if picked and variable_kind(picked, namespace) == "text"
            else None
        )
        return _dataloader_sequence(cfg, batch_size, shuffle, drop, common, has_val, vocab)
    if source == "torchvision":
        return _dataloader_torchvision(cfg, batch_size, shuffle, drop, common)
    if source == "imagefolder":
        return _dataloader_imagefolder(cfg, batch_size, shuffle, drop, common, has_val)
    return _dataloader_memory(
        cfg, batch_size, shuffle, drop, common, namespace, n_inputs, needs_targets, has_val
    )


def _checked_splits(cfg: dict, has_val: bool = True) -> tuple[float, float]:
    """(val_split, test_split), range-validated individually and TOGETHER —
    they carve from the same data, so their sum has to leave something to train
    on. Codegen enforces the rule diagnose states, because an ImageFolder tree's
    size isn't knowable pre-run: the range has to be refused here, not merely
    predicted there. A recipe whose loop never validates gets neither."""
    val = float(cfg.get("val_split", 0.0) or 0.0)
    test = float(cfg.get("test_split", 0.0) or 0.0)
    if not 0.0 <= val < 1.0:
        raise ValueError(f"val_split {val} — must be in [0, 1)")
    if not 0.0 <= test < 1.0:
        raise ValueError(f"test_split {test} — must be in [0, 1)")
    if val + test >= 1.0:
        raise ValueError(
            f"val_split {val} + test_split {test} leaves nothing to train on — they must sum below 1"
        )
    return (val, test) if has_val else (0.0, 0.0)


# The three-way carve. `test` is LAST on purpose: random_split permutes the
# indices once (from the dataset length alone) and slices by the cumulative
# lengths, so the tail — the test set — is fixed by (n, n_test, seed) and does
# NOT move when val_split changes. A held-out test set that shifted every time
# you retuned the validation fraction would be worthless.
_SPLIT_NOTE = "    # Fixed split generator: the same samples stay held out across runs/resumes."
_TEST_NOTE = (
    "    # test is carved LAST, so its membership depends only on test_split —\n"
    "    # changing val_split never moves it."
)


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

# Canonical per-sample H/W of the curated torchvision datasets (RandomCrop's
# auto-size) and their mean/std stats (Normalize). Names are validated against
# the DATA_PARAMS enum before reaching codegen.
_TV_SIZE: dict[str, int] = {
    "MNIST": 28, "FashionMNIST": 28, "KMNIST": 28, "CIFAR10": 32, "CIFAR100": 32,
}
# What every ImageNet-pretrained torchvision model was trained with. A frozen
# backbone fed differently-scaled inputs is being asked about a distribution it
# never saw, and quietly underperforms — no error, just worse numbers.
IMAGENET_STATS: tuple[tuple[float, ...], tuple[float, ...]] = (
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
)

_TV_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "MNIST": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.286,), (0.353,)),
    "KMNIST": ((0.1918,), (0.3483,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def normalize_stats(
    cfg: dict, dataset: str | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """The (mean, std) to standardize with, or None. ``dataset`` names the
    curated torchvision set whose own statistics "dataset" refers to — an image
    folder has none (nobody has measured that tree), so asking for them there is
    refused rather than silently swapped for something else.

    A legacy boolean (this param used to be a checkbox) still reads correctly:
    True meant the dataset's own statistics, which is what it keeps meaning."""
    value = cfg.get("normalize", "none")
    if isinstance(value, bool):
        value = "dataset" if value else "none"
    mode = str(value or "none")
    if mode == "none":
        return None
    if mode == "imagenet":
        return IMAGENET_STATS
    if mode == "dataset":
        if dataset is None or dataset not in _TV_STATS:
            raise ValueError(
                "normalize 'dataset' needs a curated torchvision dataset — an image folder's own "
                "statistics aren't known; use 'imagenet' (for a pretrained backbone) or 'none'"
            )
        return _TV_STATS[dataset]
    raise ValueError(f"unknown normalize '{mode}' — expected none, dataset, or imagenet")


def _compose_transforms(
    augmentations: list[str],
    resize: int | None = None,
    crop_size: int | None = None,
    normalize: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> tuple[str, str]:
    """(train_transform, eval_transform) Compose expressions. A Resize (if set) is
    deterministic and leads both. Augmentations are train-only — RandomCrop first
    (sized to the dataset via ``crop_size``, padding=4, the CIFAR standard), then
    the arg-free set in canonical order — before ToTensor; eval/val gets Resize +
    ToTensor so validation isn't perturbed. ``normalize`` (the dataset's mean/std)
    is preprocessing, not augmentation, so it lands on BOTH, after ToTensor."""
    prefix = [f"transforms.Resize(({int(resize)}, {int(resize)}))"] if resize else []
    picked = []
    if "RandomCrop" in augmentations and crop_size:
        picked.append(f"transforms.RandomCrop({int(crop_size)}, padding=4)")
    picked += [expr for name, expr in _AUGMENTATIONS if name in augmentations]
    suffix = ["transforms.ToTensor()"]
    if normalize is not None:
        mean, std = normalize
        suffix.append(f"transforms.Normalize({mean!r}, {std!r})")
    train = ", ".join([*prefix, *picked, *suffix])
    eval_ = ", ".join([*prefix, *suffix])
    return f"transforms.Compose([{train}])", f"transforms.Compose([{eval_}])"


# The Dataset-pick label reader, spliced in when a weighted sampler needs the
# class of every training sample. `.targets` is the torchvision/ImageFolder
# convention, so a big image set isn't decoded just to count labels.
_DATASET_TARGETS_SOURCE = '''def dataset_targets(dataset):
    """Class labels of a Dataset — its `.targets` when it has one (torchvision
    datasets, ImageFolder), else one pass over the samples."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[i][-1] for i in range(len(dataset))]
    return torch.as_tensor(targets).flatten().long()
'''


def _sampler_block(targets_expr: str, n_expr: str) -> list[str]:
    """Lines building a WeightedRandomSampler over the TRAINING split: each
    sample is drawn with probability inverse to its class frequency, with
    replacement so an epoch keeps its length. Rebalances what the model SEES,
    where class weights rebalance what it PAYS for a mistake."""
    return [
        f"    targets = {targets_expr}",
        "    counts = torch.bincount(targets)",
        "    sampler = WeightedRandomSampler(",
        f"        (1.0 / counts.clamp(min=1).float())[targets], num_samples={n_expr}, replacement=True",
        "    )",
    ]


def _split_and_loaders(
    val_split: float, test_split: float, *, order: str, drop: str, common: str,
    sampler_base: str | None = None,
) -> tuple[str, list[str]]:
    """(extra signature params, body lines) — the shared tail of every in-memory
    ``make_dataloaders``: carve the held-out splits, build the sampler, build the
    loaders, return them. Returns (train, val) normally and (train, val, test)
    once a test split is configured, so an unset test split leaves the existing
    two-value contract byte-identical."""
    if val_split <= 0.0 and test_split <= 0.0:
        return "", [
            *(_sampler_block(sampler_base, "len(dataset)") if sampler_base else []),
            f"    train_loader = DataLoader(dataset, batch_size=batch_size{order}{drop}{common})",
            "    return train_loader, None",
        ]

    params = f", val_split={val_split!r}"
    sizes = ["    n_val = int(len(dataset) * val_split)"]
    lengths = ["n_train", "n_val"]
    subsets = ["train_ds", "val_ds"]
    if test_split > 0.0:
        params += f", test_split={test_split!r}"
        sizes.append("    n_test = int(len(dataset) * test_split)")
        lengths.append("n_test")
        subsets.append("test_ds")
    sizes.append(f"    n_train = len(dataset) - {' - '.join(lengths[1:])}")

    lines = [
        *sizes,
        _SPLIT_NOTE,
        *([_TEST_NOTE] if test_split > 0.0 else []),
        f"    split = torch.Generator().manual_seed({SPLIT_SEED})",
        f"    {', '.join(subsets)} = random_split(dataset, [{', '.join(lengths)}], generator=split)",
        *(_sampler_block(f"{sampler_base}[train_ds.indices]", "n_train") if sampler_base else []),
        f"    train_loader = DataLoader(train_ds, batch_size=batch_size{order}{drop}{common})",
        f"    val_loader = DataLoader(val_ds, batch_size=batch_size{common}) if n_val else None",
    ]
    if test_split > 0.0:
        lines.append(f"    test_loader = DataLoader(test_ds, batch_size=batch_size{common}) if n_test else None")
    lines.append(f"    return {', '.join(sub.replace('_ds', '_loader') for sub in subsets)}")
    return params, lines


def _order_kwarg(shuffle: bool, sampler: bool) -> str:
    """The train loader's ordering argument. A sampler REPLACES shuffle (torch
    rejects both together) — it already draws in random order."""
    return ", sampler=sampler" if sampler else f", shuffle={shuffle}"


def _dataloader_memory(
    cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, namespace: dict | None,
    n_inputs: int, needs_targets: bool = True, has_val: bool = True,
) -> str:
    """In-memory source. An optionally-picked notebook variable gets the wrapping
    its *type* calls for: a DataLoader passes through, a Dataset is wrapped (with
    val_split honored — a Dataset random_splits like tensors do); a tensor/array
    pick — or no pick at all — falls back to the generic TensorDataset path
    (make_dataloaders(X, y), one X per model input)."""
    from .introspect import variable_kind

    x_var = str(cfg.get("x_var", "") or "").strip()
    kind = variable_kind(x_var, namespace) if x_var else None

    if kind == "dataloader":
        # Already a DataLoader — nothing to build; hand it straight to train().
        return "def make_dataloaders(loader):\n    return loader, None\n"
    if kind == "dataset":
        val_split, test_split = (
            _checked_splits(cfg, has_val) if needs_targets else (0.0, 0.0)
        )
        # A sampler needs labels to balance by, so it follows needs_targets.
        sampler = bool(cfg.get("weighted_sampler", False)) and needs_targets
        splitting = val_split > 0.0 or test_split > 0.0
        params, body = _split_and_loaders(
            val_split, test_split, order=_order_kwarg(shuffle, sampler), drop=drop, common=common,
            sampler_base="dataset_targets(dataset)" if sampler else None,
        )
        imports = ["import torch"] if (splitting or sampler) else []
        loaders = ["DataLoader"]
        if splitting:
            loaders.append("random_split")
        if sampler:
            loaders.append("WeightedRandomSampler")
        imports.append(f"from torch.utils.data import {', '.join(loaders)}")
        lines = [*imports, ""]
        if sampler:
            lines += ["", _DATASET_TARGETS_SOURCE.rstrip("\n")]
        lines += ["", ""]
        lines.append(f"def make_dataloaders(dataset, *, batch_size={batch_size}{params}):")
        lines += body
        return "\n".join(lines) + "\n"
    # tensors / ndarray / unknown → the TensorDataset wrapping (one X per model input).
    return _dataloader_tensors(cfg, batch_size, shuffle, drop, common, n_inputs, needs_targets, has_val)


def _dataloader_tensors(
    cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, n_inputs: int,
    needs_targets: bool = True, has_val: bool = True,
) -> str:
    """In-memory tensors → a DataLoader over a TensorDataset, with one X arg per
    model input (X for single-input, X0/X1/… for multi). With val_split > 0, a
    disjoint random_split yields a held-out val_loader too. ``needs_targets=False``
    (adversarial) drops y entirely: a loader over X alone, yielding ``(x,)``."""
    if not needs_targets:
        # Unlabeled: batches of (x,) — the GAN loop reads batch[0] as the real data.
        return (
            "import torch\n"
            "from torch.utils.data import DataLoader, TensorDataset\n\n\n"
            f"def make_dataloaders(X, *, batch_size={batch_size}):\n"
            "    dataset = TensorDataset(X)\n"
            f"    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle={shuffle}{drop}{common})\n"
            "    return train_loader, None\n"
        )
    val_split, test_split = _checked_splits(cfg, has_val)
    sampler = bool(cfg.get("weighted_sampler", False))
    xs = ["X"] if n_inputs <= 1 else [f"X{i}" for i in range(n_inputs)]
    x_params = ", ".join(xs)  # make_dataloaders params + TensorDataset args
    params, body = _split_and_loaders(
        val_split, test_split, order=_order_kwarg(shuffle, sampler), drop=drop, common=common,
        sampler_base="y.flatten().long()" if sampler else None,
    )
    loaders = ["DataLoader", "TensorDataset"]
    if val_split > 0.0 or test_split > 0.0:
        loaders.append("random_split")
    if sampler:
        loaders.append("WeightedRandomSampler")
    return "\n".join([
        "import torch",
        f"from torch.utils.data import {', '.join(loaders)}",
        "",
        "",
        f"def make_dataloaders({x_params}, y, *, batch_size={batch_size}{params}):",
        f"    dataset = TensorDataset({x_params}, y)",
        *body,
    ]) + "\n"


# The windowing Dataset, spliced into the generated loader. Slicing per item
# rather than materializing every window matters: a million tokens at block 128
# would be a gigabyte of overlapping copies.
_WINDOWS_SOURCE = '''class NextTokenWindows(Dataset):
    """Every position is an example: a window of `block_size` tokens, paired
    with the same window shifted one step — the next-token objective."""

    def __init__(self, tokens, block_size):
        self.tokens = tokens.flatten().long()
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, i):
        window = self.tokens[i : i + self.block_size + 1]
        return window[:-1], window[1:]
'''


def char_vocab(text: str) -> list[str]:
    """The character vocabulary of a text: every distinct character, sorted so
    the ids are stable — re-registering the same text gives the same encoding,
    and a stored run's ids still mean what they meant."""
    return sorted(set(text))


def _tokenizer_source(vocab: list[str]) -> str:
    """The tokenizer, baked into the generated loader. Keeping the vocabulary IN
    the source is what makes a run self-contained: the same file that encodes
    the training text decodes the model's samples, so a stored run can still be
    read back long after the notebook is gone."""
    return (
        "# The character vocabulary, built from the registered text. Ids are\n"
        "# positions in this list, so it travels with the run.\n"
        f"VOCAB = {vocab!r}\n"
        "STOI = {c: i for i, c in enumerate(VOCAB)}\n"
        "\n"
        "\n"
        "def encode(text):\n"
        '    """Text → token ids. Characters outside the vocabulary are dropped."""\n'
        "    return torch.tensor([STOI[c] for c in text if c in STOI], dtype=torch.long)\n"
        "\n"
        "\n"
        "def decode(ids):\n"
        '    """Token ids → text."""\n'
        "    return \"\".join(VOCAB[int(i)] for i in ids)\n"
    )


def _dataloader_sequence(
    cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, has_val: bool = True,
    vocab: list[str] | None = None,
) -> str:
    """A token stream → next-token windows. The held-out slices are carved
    CONTIGUOUSLY, not at random: neighbouring windows share all but one token,
    so a random split would put nearly the same text on both sides and the
    held-out loss would flatter the model into meaninglessness.

    With ``vocab`` (the pick was raw text), a character tokenizer is emitted
    alongside and make_dataloaders takes the text itself."""
    block = max(1, int(cfg.get("block_size", 128) or 128))
    val_split, test_split = _checked_splits(cfg, has_val)
    lines = [
        "import torch",
        "from torch.utils.data import DataLoader, Dataset",
        "",
        "",
    ]
    if vocab is not None:
        lines += [_tokenizer_source(vocab).rstrip("\n"), "", ""]
    lines += [_WINDOWS_SOURCE.rstrip("\n"), "", ""]
    params = f", block_size={block}"
    if val_split > 0.0:
        params += f", val_split={val_split!r}"
    if test_split > 0.0:
        params += f", test_split={test_split!r}"
    if vocab is not None:
        lines.append(f"def make_dataloaders(text, *, batch_size={batch_size}{params}):")
        lines.append("    tokens = encode(text)")
    else:
        lines.append(f"def make_dataloaders(tokens, *, batch_size={batch_size}{params}):")
        lines.append("    tokens = tokens.flatten().long()")
    if val_split > 0.0 or test_split > 0.0:
        held = []
        lines += [
            "    # Contiguous, never random: neighbouring windows share all but",
            "    # one token, so a random split would leak the training text.",
        ]
        if val_split > 0.0:
            lines.append("    n_val = int(len(tokens) * val_split)")
            held.append("n_val")
        if test_split > 0.0:
            lines.append("    n_test = int(len(tokens) * test_split)")
            held.append("n_test")
        lines.append(f"    n_train = len(tokens) - {' - '.join(held)}")
    else:
        lines.append("    n_train = len(tokens)")
    lines.append(
        f"    train_loader = DataLoader(NextTokenWindows(tokens[:n_train], block_size), "
        f"batch_size=batch_size, shuffle={shuffle}{drop}{common})"
    )
    returns = ["train_loader"]
    if val_split > 0.0:
        # A slice shorter than the window yields no examples at all — hand back
        # None rather than an empty loader the training loop would divide by.
        lines += [
            "    val_tokens = tokens[n_train:n_train + n_val]",
            f"    val_loader = DataLoader(NextTokenWindows(val_tokens, block_size), "
            f"batch_size=batch_size{common}) if len(val_tokens) > block_size else None",
        ]
        returns.append("val_loader")
    elif test_split > 0.0:
        lines.append("    val_loader = None")
        returns.append("val_loader")
    else:
        returns.append("None")
    if test_split > 0.0:
        after_val = "n_train + n_val" if val_split > 0.0 else "n_train"
        lines += [
            f"    test_tokens = tokens[{after_val}:]",
            f"    test_loader = DataLoader(NextTokenWindows(test_tokens, block_size), "
            f"batch_size=batch_size{common}) if len(test_tokens) > block_size else None",
        ]
        returns.append("test_loader")
    lines.append(f"    return {', '.join(returns)}")
    return "\n".join(lines) + "\n"


def _dataloader_torchvision(cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str) -> str:
    """A torchvision dataset → train (train=True) and val (train=False, the test
    split) DataLoaders. Train-only augmentations compose before ToTensor; val gets
    a plain ToTensor."""
    dataset = str(cfg["dataset"])
    # The dataset name lands in the source as an attribute (datasets.MNIST), so
    # unlike the repr()-escaped params it MUST be validated, not escaped — the
    # form's enum is re-checked here so a raw API caller can't inject code.
    allowed = next(p.choices for p in DATA_PARAMS if p.name == "dataset")
    if dataset not in allowed:
        raise ValueError(f"unknown torchvision dataset '{dataset}' — expected one of: {', '.join(allowed)}")
    root = str(cfg["root"])
    download = bool(cfg["download"])
    # RandomCrop sizes itself to the images the pipeline yields: the resize when
    # set, else the dataset's canonical dims. Normalize uses the canonical stats.
    resize = cfg.get("resize")
    crop_size = int(resize) if resize else _TV_SIZE[dataset]
    stats = normalize_stats(cfg, dataset)
    train_tf, eval_tf = _compose_transforms(
        list(cfg.get("augmentations") or []), resize, crop_size=crop_size, normalize=stats
    )

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


def _dataloader_imagefolder(
    cfg: dict, batch_size: int, shuffle: bool, drop: str, common: str, has_val: bool = True
) -> str:
    """A directory of class-subfolders via datasets.ImageFolder. One dataset with a
    deterministic transform (Resize + ToTensor); val_split > 0 carves a held-out
    val_loader via random_split. (Augmentations are torchvision-only — a split
    subset shares one transform, so train-only augmentation can't apply cleanly.)"""
    root = str(cfg["root"])
    val_split, test_split = _checked_splits(cfg, has_val)
    # Deterministic (train == eval); normalization is preprocessing, so it
    # applies to both — an image folder is the usual home of a fine-tuning set,
    # where matching a backbone's ImageNet statistics matters.
    transform, _ = _compose_transforms([], cfg.get("resize"), normalize=normalize_stats(cfg))
    params, body = _split_and_loaders(
        val_split, test_split, order=f", shuffle={shuffle}", drop=drop, common=common,
    )
    splitting = val_split > 0.0 or test_split > 0.0
    imports = (
        ["import torch", "from torch.utils.data import DataLoader, random_split"]
        if splitting
        else ["from torch.utils.data import DataLoader"]
    )
    return "\n".join([
        *imports,
        "from torchvision import datasets, transforms",
        "",
        "",
        f"def make_dataloaders(*, batch_size={batch_size}, root={root!r}{params}):",
        f"    transform = {transform}",
        "    dataset = datasets.ImageFolder(root, transform=transform)",
        *body,
    ]) + "\n"
