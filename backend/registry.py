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
    type: Literal["int", "float", "bool", "shape"]
    default: Any


@dataclass
class NodeDef:
    type: str
    label: str
    category: str
    color: str
    inputs: list[PinDef] = field(default_factory=list)
    outputs: list[PinDef] = field(default_factory=list)
    params: list[ParamDef] = field(default_factory=list)

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


REGISTRY: dict[str, NodeDef] = {
    "Input": NodeDef(
        type="Input", label="Input", category="io", color="#4a9eff",
        outputs=[PinDef("output", "Out")],
        params=[
            # Comma-separated dims, e.g. "1, 784" (B, F) or "1, 3, 28, 28" (B, C, H, W)
            ParamDef("shape", "Shape", "shape", "1, 784"),
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
    ),
    "Conv2d": NodeDef(
        type="Conv2d", label="Conv2d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("out_channels", "Out Channels", "int", 32),
            ParamDef("kernel_size", "Kernel Size", "int", 3),
            ParamDef("stride", "Stride", "int", 1),
            ParamDef("padding", "Padding", "int", 0),
        ],
    ),
    "ReLU": NodeDef(
        type="ReLU", label="ReLU", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
    ),
    "Sigmoid": NodeDef(
        type="Sigmoid", label="Sigmoid", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
    ),
    "Tanh": NodeDef(
        type="Tanh", label="Tanh", category="activations", color="#00bfa5",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
    ),
    "Flatten": NodeDef(
        type="Flatten", label="Flatten", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("start_dim", "Start Dim", "int", 1),
        ],
    ),
    "Dropout": NodeDef(
        type="Dropout", label="Dropout", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
        params=[
            ParamDef("p", "Dropout", "float", 0.5),
        ],
    ),
    "BatchNorm1d": NodeDef(
        type="BatchNorm1d", label="BatchNorm1d", category="layers", color="#7c4dff",
        inputs=[PinDef("input", "In")],
        outputs=[PinDef("output", "Out")],
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
    ),
}
