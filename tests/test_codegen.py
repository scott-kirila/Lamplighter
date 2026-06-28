"""Codegen string assertions — keyword args equal to their default are omitted
for cleaner generated source; non-default values are kept; positional args stay.
"""
from backend.codegen import generate_module
from tests.helpers import edge, graph, node


def _linear(params):
    g = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("lin", "Linear", params), node("out", "Output")],
        [edge("in", "lin"), edge("lin", "out")],
    )
    return generate_module(g)


def _conv(params):
    g = graph(
        [node("in", "Input", {"shape": "1, 3, 28, 28"}), node("c", "Conv2d", params), node("out", "Output")],
        [edge("in", "c"), edge("c", "out")],
    )
    return generate_module(g)


def test_omits_default_bias():
    # bias=True is the default -> dropped.
    assert "nn.Linear(784, 128)" in _linear({"out_features": 128, "bias": True})


def test_keeps_non_default_bias():
    assert "nn.Linear(784, 10, bias=False)" in _linear({"out_features": 10, "bias": False})


def test_conv_omits_default_stride_and_padding():
    code = _conv({"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0})
    assert "nn.Conv2d(3, 32, 3)" in code


def test_conv_keeps_non_default_stride_and_padding():
    code = _conv({"out_channels": 16, "kernel_size": 5, "stride": 2, "padding": 2})
    assert "nn.Conv2d(3, 16, 5, stride=2, padding=2)" in code


def test_flatten_and_dropout_default_to_bare_calls():
    g = graph(
        [
            node("in", "Input", {"shape": "1, 3, 4, 4"}),
            node("flat", "Flatten", {"start_dim": 1}),
            node("drop", "Dropout", {"p": 0.5}),
            node("out", "Output"),
        ],
        [edge("in", "flat"), edge("flat", "drop"), edge("drop", "out")],
    )
    code = generate_module(g)
    assert "nn.Flatten()" in code  # start_dim=1 is default
    assert "nn.Dropout()" in code  # p=0.5 is default
