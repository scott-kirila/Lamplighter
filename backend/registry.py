from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PinDef:
    name: str
    label: str


@dataclass
class ParamDef:
    name: str
    label: str
    type: Literal["int", "float", "bool", "shape", "enum", "tuple", "string"]
    default: Any
    choices: list[str] | None = None  # allowed values for an "enum" param
    arity: int = 2  # element count for a "tuple" param (int-or-tuple); UI hint
    optional: bool = False  # may also be None (renders/builds as None)


@dataclass
class Derived:
    """A positional constructor arg taken from the input shape, e.g. in_channels
    = input_shape[1]."""
    axis: int


@dataclass
class ModuleEmit:
    """A standard ``nn.Module`` layer. Inference instantiates it on the meta
    device; codegen renders it as ``self.layer_N = nn.<cls>(...)``. The two share
    one arg builder, so a layer is fully described by data — no per-type branch.
    """
    cls: str                                       # nn class, e.g. "Conv2d"
    # Ordered positional args: a param name, or Derived(axis) for a shape-derived
    # value. Written in call order, so derived and param args can interleave
    # (e.g. GroupNorm(num_groups, num_channels) = ["num_groups", Derived(1)]).
    pos: list[str | Derived] = field(default_factory=list)
    kw_params: list[str] = field(default_factory=list)    # params emitted as keyword args (order preserved)
    min_rank: int | None = None                    # optional input-rank precondition
    rank_msg: str | None = None                    # message for a failed rank check ("{rank}" = actual rank)
    int_input: bool = False                        # requires an integer (long) input, e.g. Embedding indices
    # Output pins -> index path into the module's return value. Default: a single
    # tensor return (path ()). Multi-output layers (LSTM returns
    # (output, (h_n, c_n))) list each pin and how to reach it.
    outputs: list[tuple[str, tuple[int, ...]]] = field(
        default_factory=lambda: [("output", ())]
    )


@dataclass
class NodeDef:
    type: str
    label: str
    category: str
    color: str
    inputs: list[PinDef] = field(default_factory=list)
    outputs: list[PinDef] = field(default_factory=list)
    params: list[ParamDef] = field(default_factory=list)
    # How this node infers shape / generates code. None for nodes with bespoke
    # handling (Input, Output, Concat). Backend-only — stripped from the API.
    emit: ModuleEmit | None = None

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


def _cast(value: Any, ptype: str) -> Any:
    if value is None:  # an optional param left unset
        return None
    if ptype == "int":
        return int(value)
    if ptype == "float":
        return float(value)
    if ptype == "bool":
        return bool(value)
    if ptype == "enum":
        return str(value)
    if ptype == "string":
        return str(value)
    if ptype == "tuple":
        # int-or-tuple: a scalar stays an int (renders/builds as e.g. 3); a list
        # becomes a tuple (renders as (3, 5)). repr() handles both.
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        return int(value)
    return value


def build_module_args(
    node_def: NodeDef, params: dict[str, Any], input_shape: list[int]
) -> tuple[list[Any], dict[str, Any]]:
    """Positional args + kwargs for a ModuleEmit node, from input-shape-derived
    values plus its params (each cast by its ParamDef type, falling back to the
    default). Shared by inference (to instantiate) and codegen (to render), so the
    two can never drift."""
    emit = node_def.emit
    assert isinstance(emit, ModuleEmit)
    pdefs = {p.name: p for p in node_def.params}
    pos: list[Any] = []
    for arg in emit.pos:
        if isinstance(arg, Derived):
            pos.append(input_shape[arg.axis])
        else:
            pd = pdefs[arg]
            pos.append(_cast(params.get(arg, pd.default), pd.type))
    kw: dict[str, Any] = {}
    for name in emit.kw_params:
        pd = pdefs[name]
        kw[name] = _cast(params.get(name, pd.default), pd.type)
    return pos, kw


def render_module_args(
    node_def: NodeDef, params: dict[str, Any], input_shape: list[int]
) -> str:
    """Source for a ModuleEmit call's args: positional values, then only the
    keyword args that differ from their default — so generated code stays minimal
    (`nn.Conv2d(3, 32, 3)` rather than `nn.Conv2d(3, 32, 3, stride=1, padding=0)`).
    Codegen-only; inference instantiates with the full args from build_module_args.
    """
    pos, kw = build_module_args(node_def, params, input_shape)
    pdefs = {p.name: p for p in node_def.params}
    # repr() so enum strings render quoted ('reflect'); ints/floats/bools are
    # byte-identical to str() either way.
    parts = [repr(a) for a in pos]
    for name, value in kw.items():
        pd = pdefs[name]
        if value == _cast(pd.default, pd.type):
            continue
        parts.append(f"{name}={value!r}")
    return ", ".join(parts)


REGISTRY: dict[str, NodeDef] = {
    "Input": NodeDef(
        type="Input", label="Input", category="io", color="#4a9eff",
        outputs=[PinDef("output", "Out")],
        params=[
            # Comma-separated dims, e.g. "1, 784" (B, F) or "1, 3, 28, 28" (B, C, H, W)
            ParamDef("shape", "Shape", "shape", "1, 784"),
            # "long" for integer index tensors (e.g. feeding an Embedding).
            ParamDef("dtype", "Dtype", "enum", "float", choices=["float", "long"]),
            # Optional forward() argument name; blank auto-names (x, or x0/x1/…).
            ParamDef("name", "Name", "string", ""),
        ],
    ),
    "Linear": NodeDef(
        type="Linear", label="Linear", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("out_features", "Out Features", "int", 128),
            ParamDef("bias", "Bias", "bool", True),
        ],
        emit=ModuleEmit("Linear", pos=[Derived(-1), "out_features"], kw_params=["bias"]),
    ),
    "Embedding": NodeDef(
        type="Embedding", label="Embedding", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("num_embeddings", "Num Embeddings", "int", 1000),
            ParamDef("embedding_dim", "Embedding Dim", "int", 64),
        ],
        # Input is a LongTensor of indices (set the Input node's dtype to "long").
        emit=ModuleEmit("Embedding", pos=["num_embeddings", "embedding_dim"], int_input=True),
    ),
    "Conv2d": NodeDef(
        type="Conv2d", label="Conv2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("out_channels", "Out Channels", "int", 32),
            ParamDef("kernel_size", "Kernel Size", "tuple", 3),
            ParamDef("stride", "Stride", "tuple", 1),
            ParamDef("padding", "Padding", "tuple", 0),
            ParamDef(
                "padding_mode", "Padding Mode", "enum", "zeros",
                choices=["zeros", "reflect", "replicate", "circular"],
            ),
        ],
        emit=ModuleEmit(
            "Conv2d",
            pos=[Derived(1), "out_channels", "kernel_size"],
            kw_params=["stride", "padding", "padding_mode"],
            min_rank=4,
            rank_msg="Conv2d expects 4D input (B,C,H,W), got {rank}D",
        ),
    ),
    "Conv1d": NodeDef(
        type="Conv1d", label="Conv1d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("out_channels", "Out Channels", "int", 32),
            ParamDef("kernel_size", "Kernel Size", "int", 3),
            ParamDef("stride", "Stride", "int", 1),
            ParamDef("padding", "Padding", "int", 0),
            ParamDef(
                "padding_mode", "Padding Mode", "enum", "zeros",
                choices=["zeros", "reflect", "replicate", "circular"],
            ),
        ],
        emit=ModuleEmit(
            "Conv1d",
            pos=[Derived(1), "out_channels", "kernel_size"],
            kw_params=["stride", "padding", "padding_mode"],
            min_rank=3,
            rank_msg="Conv1d expects 3D input (B,C,L), got {rank}D",
        ),
    ),
    "Conv3d": NodeDef(
        type="Conv3d", label="Conv3d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("out_channels", "Out Channels", "int", 32),
            ParamDef("kernel_size", "Kernel Size", "tuple", 3, arity=3),
            ParamDef("stride", "Stride", "tuple", 1, arity=3),
            ParamDef("padding", "Padding", "tuple", 0, arity=3),
            ParamDef(
                "padding_mode", "Padding Mode", "enum", "zeros",
                choices=["zeros", "reflect", "replicate", "circular"],
            ),
        ],
        emit=ModuleEmit(
            "Conv3d",
            pos=[Derived(1), "out_channels", "kernel_size"],
            kw_params=["stride", "padding", "padding_mode"],
            min_rank=5,
            rank_msg="Conv3d expects 5D input (B,C,D,H,W), got {rank}D",
        ),
    ),
    "MaxPool2d": NodeDef(
        type="MaxPool2d", label="MaxPool2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("kernel_size", "Kernel Size", "tuple", 2),
            # stride None defaults to kernel_size.
            ParamDef("stride", "Stride", "tuple", None, optional=True),
            ParamDef("padding", "Padding", "tuple", 0),
        ],
        emit=ModuleEmit(
            "MaxPool2d",
            pos=["kernel_size"],
            kw_params=["stride", "padding"],
            min_rank=4,
            rank_msg="MaxPool2d expects 4D input (B,C,H,W), got {rank}D",
        ),
    ),
    "AvgPool2d": NodeDef(
        type="AvgPool2d", label="AvgPool2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("kernel_size", "Kernel Size", "tuple", 2),
            ParamDef("stride", "Stride", "tuple", None, optional=True),
            ParamDef("padding", "Padding", "tuple", 0),
        ],
        emit=ModuleEmit(
            "AvgPool2d",
            pos=["kernel_size"],
            kw_params=["stride", "padding"],
            min_rank=4,
            rank_msg="AvgPool2d expects 4D input (B,C,H,W), got {rank}D",
        ),
    ),
    "AdaptiveAvgPool2d": NodeDef(
        type="AdaptiveAvgPool2d", label="AdaptiveAvgPool2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("output_size", "Output Size", "tuple", 1),
        ],
        emit=ModuleEmit(
            "AdaptiveAvgPool2d",
            pos=["output_size"],
            min_rank=4,
            rank_msg="AdaptiveAvgPool2d expects 4D input (B,C,H,W), got {rank}D",
        ),
    ),
    "AdaptiveMaxPool2d": NodeDef(
        type="AdaptiveMaxPool2d", label="AdaptiveMaxPool2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("output_size", "Output Size", "tuple", 1),
        ],
        emit=ModuleEmit(
            "AdaptiveMaxPool2d",
            pos=["output_size"],
            min_rank=4,
            rank_msg="AdaptiveMaxPool2d expects 4D input (B,C,H,W), got {rank}D",
        ),
    ),
    "MaxPool1d": NodeDef(
        type="MaxPool1d", label="MaxPool1d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("kernel_size", "Kernel Size", "int", 2),
            ParamDef("stride", "Stride", "int", None, optional=True),
            ParamDef("padding", "Padding", "int", 0),
        ],
        emit=ModuleEmit(
            "MaxPool1d",
            pos=["kernel_size"],
            kw_params=["stride", "padding"],
            min_rank=3,
            rank_msg="MaxPool1d expects 3D input (B,C,L), got {rank}D",
        ),
    ),
    "ReLU": NodeDef(
        type="ReLU", label="ReLU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        emit=ModuleEmit("ReLU"),
    ),
    "Sigmoid": NodeDef(
        type="Sigmoid", label="Sigmoid", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        emit=ModuleEmit("Sigmoid"),
    ),
    "Tanh": NodeDef(
        type="Tanh", label="Tanh", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        emit=ModuleEmit("Tanh"),
    ),
    "LeakyReLU": NodeDef(
        type="LeakyReLU", label="LeakyReLU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("negative_slope", "Negative Slope", "float", 0.01),
        ],
        emit=ModuleEmit("LeakyReLU", kw_params=["negative_slope"]),
    ),
    "GELU": NodeDef(
        type="GELU", label="GELU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        emit=ModuleEmit("GELU"),
    ),
    "ELU": NodeDef(
        type="ELU", label="ELU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("alpha", "Alpha", "float", 1.0),
        ],
        emit=ModuleEmit("ELU", kw_params=["alpha"]),
    ),
    "SiLU": NodeDef(
        type="SiLU", label="SiLU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        emit=ModuleEmit("SiLU"),
    ),
    "Softmax": NodeDef(
        type="Softmax", label="Softmax", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("dim", "Dim", "int", -1),
        ],
        # dim is positional so it's always emitted (nn.Softmax() warns otherwise).
        emit=ModuleEmit("Softmax", pos=["dim"]),
    ),
    "Flatten": NodeDef(
        type="Flatten", label="Flatten", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("start_dim", "Start Dim", "int", 1),
        ],
        emit=ModuleEmit("Flatten", kw_params=["start_dim"]),
    ),
    "Dropout": NodeDef(
        type="Dropout", label="Dropout", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("p", "Dropout", "float", 0.5),
        ],
        emit=ModuleEmit("Dropout", kw_params=["p"]),
    ),
    "Dropout2d": NodeDef(
        type="Dropout2d", label="Dropout2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("p", "Dropout", "float", 0.5),
        ],
        emit=ModuleEmit("Dropout2d", kw_params=["p"]),
    ),
    "BatchNorm1d": NodeDef(
        type="BatchNorm1d", label="BatchNorm1d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            # Optional[float]: None means a cumulative moving average.
            ParamDef("momentum", "Momentum", "float", 0.1, optional=True),
        ],
        emit=ModuleEmit("BatchNorm1d", pos=[Derived(-1)], kw_params=["momentum"]),
    ),
    "BatchNorm2d": NodeDef(
        type="BatchNorm2d", label="BatchNorm2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("momentum", "Momentum", "float", 0.1, optional=True),
        ],
        # num_features = channels (dim 1) of an (N, C, H, W) input.
        emit=ModuleEmit(
            "BatchNorm2d",
            pos=[Derived(1)],
            kw_params=["momentum"],
            min_rank=4,
            rank_msg="BatchNorm2d expects 4D input (N,C,H,W), got {rank}D",
        ),
    ),
    "LayerNorm": NodeDef(
        type="LayerNorm", label="LayerNorm", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        # normalized_shape = the last dim (the common case).
        emit=ModuleEmit("LayerNorm", pos=[Derived(-1)]),
    ),
    "GroupNorm": NodeDef(
        type="GroupNorm", label="GroupNorm", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("num_groups", "Num Groups", "int", 8),
        ],
        # GroupNorm(num_groups, num_channels) — the derived arg comes second.
        emit=ModuleEmit(
            "GroupNorm",
            pos=["num_groups", Derived(1)],
            min_rank=2,
            rank_msg="GroupNorm expects at least a (N, C) input, got {rank}D",
        ),
    ),
    "InstanceNorm2d": NodeDef(
        type="InstanceNorm2d", label="InstanceNorm2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        # num_features = channels (dim 1) of an (N, C, H, W) input.
        emit=ModuleEmit(
            "InstanceNorm2d",
            pos=[Derived(1)],
            min_rank=4,
            rank_msg="InstanceNorm2d expects 4D input (N,C,H,W), got {rank}D",
        ),
    ),
    # Recurrent layers — multi-output: a sequence `output` plus hidden state(s).
    # input_size = the last dim; input is 3D (seq, batch, features), or
    # (batch, seq, features) with batch_first.
    "RNN": NodeDef(
        type="RNN", label="RNN", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out"), PinDef("h_n", "h_n")],
        params=[
            ParamDef("hidden_size", "Hidden Size", "int", 128),
            ParamDef("num_layers", "Num Layers", "int", 1),
            ParamDef("nonlinearity", "Nonlinearity", "enum", "tanh", choices=["tanh", "relu"]),
            ParamDef("batch_first", "Batch First", "bool", False),
            ParamDef("bidirectional", "Bidirectional", "bool", False),
        ],
        emit=ModuleEmit(
            "RNN",
            pos=[Derived(-1), "hidden_size"],
            kw_params=["num_layers", "nonlinearity", "batch_first", "bidirectional"],
            outputs=[("output", (0,)), ("h_n", (1,))],
            min_rank=3,
            rank_msg="RNN expects 3D input (seq, batch, features), got {rank}D",
        ),
    ),
    "LSTM": NodeDef(
        type="LSTM", label="LSTM", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out"), PinDef("h_n", "h_n"), PinDef("c_n", "c_n")],
        params=[
            ParamDef("hidden_size", "Hidden Size", "int", 128),
            ParamDef("num_layers", "Num Layers", "int", 1),
            ParamDef("batch_first", "Batch First", "bool", False),
            ParamDef("bidirectional", "Bidirectional", "bool", False),
        ],
        emit=ModuleEmit(
            "LSTM",
            pos=[Derived(-1), "hidden_size"],
            kw_params=["num_layers", "batch_first", "bidirectional"],
            outputs=[("output", (0,)), ("h_n", (1, 0)), ("c_n", (1, 1))],
            min_rank=3,
            rank_msg="LSTM expects 3D input (seq, batch, features), got {rank}D",
        ),
    ),
    "GRU": NodeDef(
        type="GRU", label="GRU", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out"), PinDef("h_n", "h_n")],
        params=[
            ParamDef("hidden_size", "Hidden Size", "int", 128),
            ParamDef("num_layers", "Num Layers", "int", 1),
            ParamDef("batch_first", "Batch First", "bool", False),
            ParamDef("bidirectional", "Bidirectional", "bool", False),
        ],
        emit=ModuleEmit(
            "GRU",
            pos=[Derived(-1), "hidden_size"],
            kw_params=["num_layers", "batch_first", "bidirectional"],
            outputs=[("output", (0,)), ("h_n", (1,))],
            min_rank=3,
            rank_msg="GRU expects 3D input (seq, batch, features), got {rank}D",
        ),
    ),
    "Concat": NodeDef(
        type="Concat", label="Concat", category="ops", color="#ffa726",
        inputs=[PinDef("in0", "In 0"), PinDef("in1", "In 1")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("dim", "Dim", "int", 1),
        ],
    ),
    "Output": NodeDef(
        type="Output", label="Output", category="io", color="#ff6b6b",
        inputs=[PinDef("input", "In")],
        params=[
            # Optional return name; when a multi-output model names any Output, it
            # returns a namedtuple (blank fields auto-name out0/out1/…).
            ParamDef("name", "Name", "string", ""),
        ],
    ),
}


# Graph-global training config — rendered by the same param controls as nodes.
TRAINING_PARAMS: list[ParamDef] = [
    ParamDef(
        "loss", "Loss", "enum", "CrossEntropyLoss",
        choices=["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "NLLLoss", "L1Loss"],
    ),
    ParamDef(
        "optimizer", "Optimizer", "enum", "Adam",
        choices=["Adam", "AdamW", "SGD", "RMSprop"],
    ),
    ParamDef("lr", "Learning Rate", "float", 1e-3),
    ParamDef("weight_decay", "Weight Decay", "float", 0.0),
    ParamDef("epochs", "Epochs", "int", 10),
    ParamDef("batch_size", "Batch Size", "int", 32),
    # Fraction of the data held out for validation each run (0 = none).
    ParamDef("val_split", "Validation Split", "float", 0.0),
    # Top-1 accuracy is reported only for classification losses (see codegen).
    ParamDef("metric", "Metric", "enum", "accuracy", choices=["accuracy", "none"]),
    # Baseline choices; the API replaces these with the live available_devices().
    ParamDef("device", "Device", "enum", "auto", choices=["auto", "cpu"]),
]


def default_training() -> dict[str, Any]:
    return {p.name: p.default for p in TRAINING_PARAMS}


def available_devices() -> list[str]:
    """Devices the *installed* torch actually supports, so the training form only
    offers what will work in this kernel — a CPU-only build won't list cuda/mps,
    and an older torch without the mps backend is handled gracefully. "auto"
    resolves at train time; "cpu" is always available."""
    import torch

    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    return devices
