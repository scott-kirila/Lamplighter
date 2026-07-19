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


def test_graph_issues_allows_multiple_inputs():
    # Multiple Input nodes are fine — each becomes a forward() argument.
    assert _issues(["Input", "Input", "Output"]) == []


def test_graph_issues_allows_multiple_outputs():
    # Multiple Output nodes are fine — the model returns a tuple of them.
    assert _issues(["Input", "Output", "Output"]) == []


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
    assert shapes[("bn", "output")] == [1, 16]


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
    assert shapes[("cat", "output")] == [1, 10]


# --- parameter counts -------------------------------------------------------

def test_param_counts_collected_during_inference():
    g = graph(
        [
            node("in", "Input", {"shape": "1, 784"}),
            node("a", "Linear", {"out_features": 128}),
            node("r", "ReLU"),
            node("b", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [edge("in", "a"), edge("a", "r"), edge("r", "b"), edge("b", "out")],
    )
    counts: dict = {}
    _, errors = infer_shapes(g, param_counts=counts)
    assert errors == {}
    assert counts["a"]["count"] == 784 * 128 + 128   # weight + bias
    assert counts["a"]["terms"] == [[128, 784], [128]]  # the factorization
    assert counts["b"]["count"] == 128 * 10 + 10
    assert counts["r"] == {"count": 0, "terms": []}  # activations are parameter-free
    assert "in" not in counts and "out" not in counts  # not nn modules


def test_param_counts_ride_the_ws_payload():
    from fastapi.testclient import TestClient

    from backend.app import app

    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 4}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    from tests.helpers import single_model_project

    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "validate", "project": single_model_project(g).model_dump()})
            msg = ws.receive_json()
    assert msg["type"] == "shapes"
    # Results are keyed per model ("model" is the sole model's id).
    assert msg["models"]["model"]["params"]["l"] == {"count": 8 * 4 + 4, "terms": [[4, 8], [4]]}
