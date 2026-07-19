"""Multi-return models: several wired Output nodes → the model returns a tuple,
ordered top-to-bottom by canvas position. A single Output is unchanged."""
import pytest
import torch

from lamplighter.backend.codegen import generate_module
from lamplighter.backend.inference import infer_shapes
from tests.helpers import edge, graph, node


def test_single_output_returns_one_value():
    g = graph(
        [node("in", "Input", {"shape": "4, 64"}), node("l", "Linear", {"out_features": 10}), node("o", "Output")],
        [edge("in", "l"), edge("l", "o")],
    )
    code = generate_module(g)
    assert code.rstrip().endswith("return t0")  # no tuple for a single output


def test_two_outputs_return_a_tuple_ordered_by_position():
    # l1 feeds Output o1 (top) directly; l1 -> l2 feeds Output o2 (below).
    g = graph(
        [
            node("in", "Input", {"shape": "4, 64"}),
            node("l1", "Linear", {"out_features": 10}),
            node("l2", "Linear", {"out_features": 5}),
            node("o1", "Output", y=0),
            node("o2", "Output", y=100),
        ],
        [edge("in", "l1"), edge("l1", "o1"), edge("l1", "l2"), edge("l2", "o2")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}

    code = generate_module(g)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    model = ns["GeneratedModel"]().eval()
    out = model(torch.randn(4, 64))

    assert isinstance(out, tuple) and len(out) == 2
    # o1 (y=0) first -> Linear1 [4, 10]; o2 (y=100) -> Linear2 [4, 5].
    assert list(out[0].shape) == [4, 10]
    assert list(out[1].shape) == [4, 5]


def test_unwired_output_is_ignored():
    # A stray, unconnected Output must not break codegen (like any stray node).
    g = graph(
        [
            node("in", "Input", {"shape": "4, 64"}),
            node("l", "Linear", {"out_features": 10}),
            node("o", "Output"),
            node("stray", "Output"),  # no input wired
        ],
        [edge("in", "l"), edge("l", "o")],
    )
    code = generate_module(g)
    assert code.rstrip().endswith("return t0")  # only the wired output counts


def test_no_connected_output_raises():
    g = graph(
        [node("in", "Input", {"shape": "4, 64"}), node("o", "Output")],
        [],  # Output not wired
    )
    with pytest.raises(ValueError):
        generate_module(g)
