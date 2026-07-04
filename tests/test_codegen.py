"""Codegen string assertions — keyword args equal to their default are omitted
for cleaner generated source; non-default values are kept; positional args stay.
"""
import pytest

from backend.codegen import class_name_for, generate_module, sanitize_class_name
from tests.helpers import edge, graph, node


def test_sanitize_class_name():
    assert sanitize_class_name("Generator") == "Generator"
    assert sanitize_class_name("Discriminator") == "Discriminator"
    assert sanitize_class_name("Model 2") == "Model2"
    assert sanitize_class_name("my-gan") == "MyGan"
    assert sanitize_class_name("123") == "Model123"  # can't start with a digit
    assert sanitize_class_name("") == "Model"  # nothing usable → a sane default


def test_class_name_for_keeps_generatedmodel_for_a_lone_model():
    # A sole model stays the classic name (byte-identical single-model output);
    # only when models coexist does each take its own sanitized name.
    assert class_name_for("Model", sole=True) == "GeneratedModel"
    assert class_name_for("Generator", sole=True) == "GeneratedModel"
    assert class_name_for("Generator", sole=False) == "Generator"


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


def test_softmax_always_emits_dim():
    # dim is positional, so it's always present even at the default (a bare
    # nn.Softmax() warns about an implicit dim).
    g = graph(
        [node("in", "Input", {"shape": "1, 10"}), node("sm", "Softmax"), node("out", "Output")],
        [edge("in", "sm"), edge("sm", "out")],
    )
    assert "nn.Softmax(-1)" in generate_module(g)


def test_groupnorm_derived_arg_order():
    # GroupNorm(num_groups, num_channels): the derived num_channels comes SECOND.
    g = graph(
        [
            node("in", "Input", {"shape": "1, 16, 4, 4"}),
            node("gn", "GroupNorm", {"num_groups": 4}),
            node("out", "Output"),
        ],
        [edge("in", "gn"), edge("gn", "out")],
    )
    code = generate_module(g)
    assert "nn.GroupNorm(4, 16)" in code  # num_groups=4, num_channels=16


def test_enum_default_omitted():
    # padding_mode default 'zeros' -> dropped.
    code = _conv({"out_channels": 32, "kernel_size": 3, "padding_mode": "zeros"})
    assert "nn.Conv2d(3, 32, 3)" in code
    assert "padding_mode" not in code


def test_enum_non_default_rendered_quoted():
    code = _conv({"out_channels": 32, "kernel_size": 3, "padding_mode": "reflect"})
    assert "padding_mode='reflect'" in code


def test_tuple_scalar_renders_bare():
    # kernel_size 3 (scalar) stays 3, not (3, 3).
    assert "nn.Conv2d(3, 32, 3)" in _conv({"out_channels": 32, "kernel_size": 3})


def test_tuple_array_renders_as_tuple():
    code = _conv({"out_channels": 16, "kernel_size": [3, 5]})
    assert "nn.Conv2d(3, 16, (3, 5))" in code


def test_tuple_keyword_kept_when_non_default():
    code = _conv({"out_channels": 16, "kernel_size": 3, "stride": [2, 1]})
    assert "stride=(2, 1)" in code


def test_stray_node_does_not_break_codegen():
    # A disconnected scratch node (no input) must not break the wired model.
    g = graph(
        [
            node("in", "Input", {"shape": "1, 784"}),
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
            node("stray", "ReLU"),  # dangling, has "no input connected"
        ],
        [edge("in", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)
    assert "nn.Linear(784, 10)" in code
    assert "nn.ReLU" not in code  # stray ReLU not emitted


def test_stray_input_node_ignored():
    g = graph(
        [
            node("in", "Input", {"shape": "1, 784"}),
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
            node("stray_in", "Input", {"shape": "1, 5"}),  # extra, unconnected
        ],
        [edge("in", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)  # must not raise "expected exactly 1 Input"
    assert "nn.Linear(784, 10)" in code


def test_error_in_model_path_still_raises():
    # An error on a node that DOES feed the Output is still fatal.
    g = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("c", "Conv2d"), node("out", "Output")],
        [edge("in", "c"), edge("c", "out")],
    )
    with pytest.raises(ValueError):
        generate_module(g)


def _batchnorm(params):
    g = graph(
        [node("in", "Input", {"shape": "8, 16"}), node("bn", "BatchNorm1d", params), node("out", "Output")],
        [edge("in", "bn"), edge("bn", "out")],
    )
    return generate_module(g)


def test_optional_default_omitted():
    # momentum default 0.1 -> dropped.
    code = _batchnorm({"momentum": 0.1})
    assert "nn.BatchNorm1d(16)" in code
    assert "momentum" not in code


def test_optional_none_rendered():
    code = _batchnorm({"momentum": None})
    assert "momentum=None" in code


def test_optional_value_rendered():
    code = _batchnorm({"momentum": 0.05})
    assert "momentum=0.05" in code


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
