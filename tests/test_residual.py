"""The Add node — element-wise sum, the residual/skip-connection primitive.

Inference must follow torch's own broadcasting rule (so shapes can't disagree
with the generated `a + b`), codegen must emit a plain `+`, and a real skip
connection must behave like one at runtime: with the transform branch zeroed,
the block is the identity.
"""
import torch

from lamplighter.backend.codegen import exec_generated, generate_module
from lamplighter.backend.inference import infer_shapes, primary_shapes
from tests.helpers import edge, graph, node


def _residual_block():
    """x → Linear(64) → ReLU → Linear(64) ┐
       x ────────────────────────────────┴ Add → Output"""
    return graph(
        [node("in", "Input", {"shape": "8, 64"}),
         node("l1", "Linear", {"out_features": 64}), node("r", "ReLU"),
         node("l2", "Linear", {"out_features": 64}),
         node("add", "Add"), node("out", "Output")],
        [edge("in", "l1"), edge("l1", "r"), edge("r", "l2"),
         edge("l2", "add", tgt_h="in0"), edge("in", "add", tgt_h="in1"),
         edge("add", "out")],
    )


# --- shape inference ------------------------------------------------------------

def test_add_preserves_the_shape_of_equal_inputs():
    g = _residual_block()
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["add"] == [8, 64]


def test_add_follows_torch_broadcasting():
    # (8, 64) + (8, 1) broadcasts to (8, 64) — torch's rule, verbatim.
    g = graph(
        [node("a", "Input", {"shape": "8, 64"}), node("b", "Input", {"shape": "8, 1"}),
         node("add", "Add"), node("out", "Output")],
        [edge("a", "add", tgt_h="in0"), edge("b", "add", tgt_h="in1"), edge("add", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["add"] == [8, 64]


def test_add_flags_unbroadcastable_shapes():
    g = graph(
        [node("a", "Input", {"shape": "8, 64"}), node("b", "Input", {"shape": "8, 100"}),
         node("add", "Add"), node("out", "Output")],
        [edge("a", "add", tgt_h="in0"), edge("b", "add", tgt_h="in1"), edge("add", "out")],
    )
    _, errors = infer_shapes(g)
    assert "cannot add shapes" in errors["add"]
    assert "not broadcastable" in errors["add"]


def test_add_with_one_wire_is_flagged():
    g = graph(
        [node("a", "Input", {"shape": "8, 64"}), node("add", "Add"), node("out", "Output")],
        [edge("a", "add", tgt_h="in0"), edge("add", "out")],
    )
    _, errors = infer_shapes(g)
    assert "≥2 inputs" in errors["add"]


# --- codegen + runtime -----------------------------------------------------------

def test_residual_block_generates_a_plain_sum_and_runs():
    g = _residual_block()
    src = generate_module(g)
    # The skip reads as ordinary Python — `t2 = t1 + x` (arg order by handle).
    assert " + " in src and "torch.add" not in src

    model = next(
        v for v in exec_generated(src, "<test-residual>").values()
        if isinstance(v, type) and issubclass(v, torch.nn.Module)
    )()
    x = torch.randn(8, 64)
    out = model(x)
    assert tuple(out.shape) == (8, 64)  # matches what inference predicted

    # Zero the transform branch: a true skip connection is now the identity.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    assert torch.equal(model(x), x)
