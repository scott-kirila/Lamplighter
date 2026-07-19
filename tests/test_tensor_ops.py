"""The OpEmit tensor ops — Reshape, Permute, Mean.

One declarative mechanism (a template expression, values canonicalized by param
type): inference evals the rendered string on a meta tensor, codegen splices
the identical string into forward() — torch itself is the shape rule, so the
two can't disagree.
"""
import pytest
import torch

from lamplighter.backend.codegen import exec_generated, generate_module
from lamplighter.backend.inference import infer_shapes, primary_shapes
from lamplighter.backend.registry import REGISTRY, render_op
from tests.helpers import edge, graph, node


def _seq(mid, in_shape="8, 5, 16"):
    return graph(
        [node("in", "Input", {"shape": in_shape}), mid, node("out", "Output")],
        [edge("in", "m"), edge("m", "out")],
    )


def _model_from(src):
    found = None
    for v in exec_generated(src, "<test-ops>").values():
        if isinstance(v, type) and issubclass(v, torch.nn.Module) and v is not torch.nn.Module:
            found = v
    return found()


# --- shape inference (torch's own rule, via the meta eval) -----------------------

def test_reshape_reshapes_the_non_batch_dims():
    g = _seq(node("m", "Reshape", {"shape": "1, 28, 28"}), in_shape="8, 784")
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["m"] == [8, 28, 28]  # batch rides through


def test_permute_reorders_dims():
    g = _seq(node("m", "Permute", {"dims": "0, 2, 1"}))
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["m"] == [8, 16, 5]


def test_mean_pools_a_dim():
    g = _seq(node("m", "Mean", {"dim": 1}))
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["m"] == [8, 16]

    g = _seq(node("m", "Mean", {"dim": 1, "keepdim": True}))
    shapes, _ = infer_shapes(g)
    assert primary_shapes(g, shapes)["m"] == [8, 1, 16]


def test_op_errors_come_from_torch_and_land_on_the_node():
    # Element-count mismatch: 784 ≠ 27×27.
    _, errors = infer_shapes(_seq(node("m", "Reshape", {"shape": "1, 27, 27"}), in_shape="8, 784"))
    assert "m" in errors and "shape" in errors["m"].lower()
    # Wrong permutation arity for a 3D input.
    _, errors = infer_shapes(_seq(node("m", "Permute", {"dims": "0, 1"})))
    assert "m" in errors


def test_hostile_param_strings_cannot_reach_the_template():
    # Comma-int canonicalization is the injection guard for spliced int lists.
    with pytest.raises(ValueError, match="comma-separated integers"):
        render_op(REGISTRY["Permute"], {"dims": "0); import os #"}, "x")
    with pytest.raises(ValueError, match="comma-separated integers"):
        render_op(REGISTRY["Reshape"], {"shape": "1, 28); os.system('x')"}, "x")


# --- codegen: the identical expression, spliced ----------------------------------

def test_ops_generate_plain_tensor_expressions():
    g = graph(
        [node("in", "Input", {"shape": "8, 784"}),
         node("r", "Reshape", {"shape": "1, 28, 28"}),
         node("p", "Permute", {"dims": "0, 2, 1"}),
         node("mn", "Mean", {"dim": 2}),
         node("out", "Output")],
        [edge("in", "r"), edge("r", "p"), edge("p", "mn"), edge("mn", "out")],
    )
    src = generate_module(g)
    assert "t0 = x.reshape(x.size(0), 28, 28)" in src
    assert "t1 = t0.permute(0, 2, 1)" in src
    assert "t2 = t1.mean(dim=2, keepdim=False)" in src

    out = _model_from(src)(torch.randn(8, 784))
    shapes, _ = infer_shapes(g)
    assert list(out.shape) == primary_shapes(g, shapes)["mn"]  # runtime agrees


def test_transformer_with_a_mean_pool_head():
    # The op the plan was for: tokens → Embedding → Transformer Block →
    # Mean over the sequence → Linear head. No Flatten hack.
    g = graph(
        [node("in", "Input", {"shape": "8, 5", "dtype": "long"}),
         node("emb", "Embedding", {"num_embeddings": 100, "embedding_dim": 16}),
         node("tb", "TransformerEncoderLayer", {"nhead": 4, "dim_feedforward": 32}),
         node("pool", "Mean", {"dim": 1}),
         node("cls", "Linear", {"out_features": 10}),
         node("out", "Output")],
        [edge("in", "emb"), edge("emb", "tb"), edge("tb", "pool"),
         edge("pool", "cls"), edge("cls", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["pool"] == [8, 16]  # (N, S, E) → (N, E)
    assert primary_shapes(g, shapes)["cls"] == [8, 10]

    model = _model_from(generate_module(g)).eval()
    out = model(torch.randint(0, 100, (8, 5)))
    assert tuple(out.shape) == (8, 10)
