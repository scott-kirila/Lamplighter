"""Tier 2 — graph-level validation and inference error paths.

Pins the exact messages the editor surfaces, so a refactor can't silently reword
or drop them.
"""
from backend.inference import graph_issues, infer_shapes
from tests.helpers import edge, graph, node


# --- graph_issues truth table ---------------------------------------------

def _issues(types):
    g = graph([node(str(i), t) for i, t in enumerate(types)], [])
    return graph_issues(g)


def test_graph_issues_clean():
    assert _issues(["Input", "Output"]) == []


def test_graph_issues_empty_canvas_is_silent():
    assert _issues([]) == []


def test_graph_issues_missing_output():
    assert _issues(["Input", "Linear"]) == ["No Output node — add one to mark the model's result."]


def test_graph_issues_missing_input():
    assert _issues(["Linear", "Output"]) == ["No Input node — add one to define the model's input."]


def test_graph_issues_duplicate_input():
    assert _issues(["Input", "Input", "Output"]) == ["2 Input nodes — only one is supported."]


def test_graph_issues_duplicate_output():
    assert _issues(["Input", "Output", "Output"]) == ["2 Output nodes — only one is supported."]


# --- inference error paths -------------------------------------------------

def test_no_input_connected():
    g = graph([node("lin", "Linear"), node("out", "Output")], [edge("lin", "out")])
    _, errors = infer_shapes(g)
    assert errors["lin"] == "no input connected"


def test_cycle_detected():
    g = graph(
        [node("a", "ReLU"), node("b", "ReLU")],
        [edge("a", "b"), edge("b", "a")],
    )
    _, errors = infer_shapes(g)
    assert "cycle detected" in errors.values()


def test_conv_rank_error_message():
    g = graph(
        [node("in", "Input", {"shape": "1, 784"}), node("conv", "Conv2d"), node("out", "Output")],
        [edge("in", "conv"), edge("conv", "out")],
    )
    _, errors = infer_shapes(g)
    assert errors["conv"] == "Conv2d expects 4D input (B,C,H,W), got 2D"


def test_concat_rank_mismatch():
    g = graph(
        [
            node("a", "Input", {"shape": "1, 4"}),
            node("b", "Input", {"shape": "1, 2, 3"}),
            node("cat", "Concat", {"dim": 1}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"), edge("cat", "out")],
    )
    _, errors = infer_shapes(g)
    assert errors["cat"] == "rank mismatch between inputs"


def test_concat_size_mismatch():
    g = graph(
        [
            node("a", "Input", {"shape": "1, 4"}),
            node("b", "Input", {"shape": "2, 6"}),
            node("cat", "Concat", {"dim": 1}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"), edge("cat", "out")],
    )
    _, errors = infer_shapes(g)
    assert errors["cat"] == "size mismatch on dim 0: [1, 2]"


def test_embedding_rejects_float_input():
    # A default (float) Input into Embedding is caught in the editor, not at
    # runtime — meta skips the dtype check, so inference enforces it explicitly.
    g = graph(
        [
            node("in", "Input", {"shape": "1, 10"}),  # default dtype = float
            node("emb", "Embedding", {"num_embeddings": 100, "embedding_dim": 16}),
            node("out", "Output"),
        ],
        [edge("in", "emb"), edge("emb", "out")],
    )
    _, errors = infer_shapes(g)
    assert "integer index input" in errors.get("emb", "")


def test_batchnorm_batch_size_one_resolves():
    # eval-mode shape inference doesn't trip BatchNorm's train-time batch check.
    g = graph(
        [node("in", "Input", {"shape": "1, 16"}), node("bn", "BatchNorm1d"), node("out", "Output")],
        [edge("in", "bn"), edge("bn", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert shapes["bn"] == [1, 16]


def test_concat_happy_path():
    g = graph(
        [
            node("a", "Input", {"shape": "1, 4"}),
            node("b", "Input", {"shape": "1, 6"}),
            node("cat", "Concat", {"dim": 1}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"), edge("cat", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert shapes["cat"] == [1, 10]
