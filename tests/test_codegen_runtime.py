"""Tier 1 #2 — the strongest correctness check.

Generated source must exec, instantiate, and — when run on a real tensor of the
Input shape — produce an output whose shape matches what `infer_shapes` predicted.
This validates codegen, inference, AND their agreement against real PyTorch in one
shot, so shape bugs in new nodes surface here for free.

Only single-Input graphs apply (codegen targets one `forward(self, x)` arg);
multi-input ops like Concat are covered by the inference tests instead.
"""
import pytest
import torch

from backend.codegen import generate_module
from backend.inference import infer_shapes
from tests.helpers import edge, graph, node, output_id


def _mlp():
    # Batch size > 1 so BatchNorm1d is valid.
    g = graph(
        [
            node("in", "Input", {"shape": "8, 784"}),
            node("l1", "Linear", {"out_features": 128, "bias": True}),
            node("relu", "ReLU"),
            node("drop", "Dropout", {"p": 0.5}),
            node("bn", "BatchNorm1d"),
            node("l2", "Linear", {"out_features": 10, "bias": False}),
            node("out", "Output"),
        ],
        [
            edge("in", "l1"), edge("l1", "relu"), edge("relu", "drop"),
            edge("drop", "bn"), edge("bn", "l2"), edge("l2", "out"),
        ],
    )
    return g, [8, 784]


def _cnn():
    g = graph(
        [
            node("in", "Input", {"shape": "1, 3, 28, 28"}),
            node("conv", "Conv2d", {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0}),
            node("relu", "ReLU"),
            node("flat", "Flatten", {"start_dim": 1}),
            node("lin", "Linear", {"out_features": 10, "bias": True}),
            node("out", "Output"),
        ],
        [
            edge("in", "conv"), edge("conv", "relu"),
            edge("relu", "flat"), edge("flat", "lin"), edge("lin", "out"),
        ],
    )
    return g, [1, 3, 28, 28]


CASES = {"mlp": _mlp(), "cnn": _cnn()}


@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_generated_model_runs_and_matches_inferred_shape(case):
    g, input_shape = case
    shapes, errors = infer_shapes(g)
    assert errors == {}

    code = generate_module(g)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - exercising our own generated source
    # eval() so BatchNorm1d/Dropout behave with batch size 1.
    model = ns["GeneratedModel"]().eval()

    out = model(torch.zeros(input_shape))
    assert list(out.shape) == shapes[output_id(g)]
