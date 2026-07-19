"""Attention nodes: Self-Attention (nn.MultiheadAttention, q = k = v) and the
Transformer Block (nn.TransformerEncoderLayer) — both pure registry data, with
the one generic addition of ModuleEmit.call_repeat (the module is called with
its input repeated, so self-attention renders as `self.layer_N(x, x, x)`).
"""
import torch

from lamplighter.backend.codegen import exec_generated, generate_module
from lamplighter.backend.inference import infer_shapes, primary_shapes
from tests.helpers import edge, graph, node


def _seq_graph(mid_node):
    return graph(
        [node("in", "Input", {"shape": "8, 5, 16"}), mid_node, node("out", "Output")],
        [edge("in", "m"), edge("m", "out")],
    )


def _build(src):
    return next(
        v for v in exec_generated(src, "<test-attn>").values()
        if isinstance(v, type) and issubclass(v, torch.nn.Module)
    )()


# --- shape inference -------------------------------------------------------------

def test_self_attention_preserves_the_sequence_shape():
    g = _seq_graph(node("m", "MultiheadAttention", {"num_heads": 8}))
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["m"] == [8, 5, 16]


def test_transformer_block_preserves_the_sequence_shape():
    g = _seq_graph(node("m", "TransformerEncoderLayer", {"nhead": 8, "dim_feedforward": 32}))
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["m"] == [8, 5, 16]


def test_indivisible_heads_surface_as_a_node_error():
    # embed 16 with 3 heads — torch's own constructor error lands on the node.
    g = _seq_graph(node("m", "MultiheadAttention", {"num_heads": 3}))
    _, errors = infer_shapes(g)
    assert "m" in errors and "divisible" in errors["m"]


def test_attention_requires_a_3d_input():
    g = graph(
        [node("in", "Input", {"shape": "8, 16"}),
         node("m", "MultiheadAttention", {}), node("out", "Output")],
        [edge("in", "m"), edge("m", "out")],
    )
    _, errors = infer_shapes(g)
    assert "3D input" in errors["m"]


# --- codegen + runtime -----------------------------------------------------------

def test_self_attention_renders_qkv_and_runs():
    g = _seq_graph(node("m", "MultiheadAttention", {"num_heads": 4}))
    src = generate_module(g)
    # q = k = v = the input, and the (output, weights) tuple is unpacked.
    assert "nn.MultiheadAttention(16, 4, batch_first=True)" in src
    assert "(x, x, x)" in src

    out = _build(src)(torch.randn(8, 5, 16))
    assert tuple(out.shape) == (8, 5, 16)


def test_tiny_transformer_classifier_builds_and_runs():
    # tokens → Embedding → Transformer Block → Flatten → Linear(10): the era's
    # default architecture, straight off the canvas.
    g = graph(
        [node("in", "Input", {"shape": "8, 5", "dtype": "long"}),
         node("emb", "Embedding", {"num_embeddings": 100, "embedding_dim": 16}),
         node("tb", "TransformerEncoderLayer",
              {"nhead": 4, "dim_feedforward": 32, "norm_first": True}),
         node("f", "Flatten"), node("cls", "Linear", {"out_features": 10}),
         node("out", "Output")],
        [edge("in", "emb"), edge("emb", "tb"), edge("tb", "f"),
         edge("f", "cls"), edge("cls", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["cls"] == [8, 10]

    src = generate_module(g)
    assert "norm_first=True" in src
    model = _build(src).eval()  # eval: dropout off for a deterministic check
    out = model(torch.randint(0, 100, (8, 5)))
    assert tuple(out.shape) == (8, 10)  # runtime agrees with inference
