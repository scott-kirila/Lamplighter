"""Multi-input models: several Input nodes become the forward() arguments
(x0, x1, … ordered by canvas position). A single Input keeps the plain
forward(self, x) form, and the generated train() adapts to a tuple of tensors."""
import torch

from backend.codegen import generate_module, generate_training
from tests.helpers import edge, graph, node


def _two_input_graph():
    # Two Inputs concatenated on the feature dim, then a Linear head.
    return graph(
        [
            node("a", "Input", {"shape": "4, 8"}, y=0),
            node("b", "Input", {"shape": "4, 8"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [
            edge("a", "cat", tgt_h="in0"),
            edge("b", "cat", tgt_h="in1"),
            edge("cat", "lin"),
            edge("lin", "out"),
        ],
    )


def test_two_inputs_generate_two_forward_args():
    code = generate_module(_two_input_graph())
    assert "def forward(self, x0, x1):" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    model = ns["GeneratedModel"]().eval()
    out = model(torch.randn(4, 8), torch.randn(4, 8))
    assert list(out.shape) == [4, 10]


def test_single_input_is_unchanged():
    g = graph(
        [node("in", "Input", {"shape": "4, 8"}), node("lin", "Linear", {"out_features": 10}), node("out", "Output")],
        [edge("in", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)
    assert "def forward(self, x):" in code  # lone input stays `x`, not `x0`


def test_arg_order_follows_canvas_position():
    # `top` (y=0) must bind to x0, `bot` (y=100) to x1. The two branches expect
    # different feature sizes, so a swapped order would raise at forward time —
    # a clean, unambiguous check of the position→arg mapping.
    g = graph(
        [
            node("top", "Input", {"shape": "4, 8"}, y=0),
            node("bot", "Input", {"shape": "4, 5"}, y=100),
            node("lt", "Linear", {"out_features": 6}),
            node("lb", "Linear", {"out_features": 6}),
            node("cat", "Concat", {"dim": 1}),
            node("head", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [
            edge("top", "lt"),
            edge("bot", "lb"),
            edge("lt", "cat", tgt_h="in0"),
            edge("lb", "cat", tgt_h="in1"),
            edge("cat", "head"),
            edge("head", "out"),
        ],
    )
    code = generate_module(g)
    assert "def forward(self, x0, x1):" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    model = ns["GeneratedModel"]().eval()
    # x0 -> top (8 features), x1 -> bot (5 features); wrong order shape-mismatches.
    out = model(torch.randn(4, 8), torch.randn(4, 5))
    assert list(out.shape) == [4, 10]


def test_unconnected_input_is_excluded():
    # A stray Input wired to nothing must not become a forward argument.
    g = graph(
        [
            node("in", "Input", {"shape": "4, 8"}),
            node("stray", "Input", {"shape": "4, 3"}),  # not wired anywhere
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [edge("in", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)
    assert "def forward(self, x):" in code  # only the wired input counts


def test_training_adapts_to_multiple_inputs():
    g = _two_input_graph()
    train_code = generate_training(g)
    assert "def train(model, Xs, y" in train_code

    mod_ns: dict = {}
    exec(generate_module(g), mod_ns)  # noqa: S102
    model = mod_ns["GeneratedModel"]()

    tr_ns: dict = {}
    exec(train_code, tr_ns)  # noqa: S102
    X0, X1 = torch.randn(16, 8), torch.randn(16, 8)
    y = torch.randint(0, 10, (16,))
    tr_ns["train"](model, (X0, X1), y, epochs=1, batch_size=8)  # runs a real forward/backward


def test_training_single_input_signature_unchanged():
    g = graph(
        [node("in", "Input", {"shape": "4, 8"}), node("lin", "Linear", {"out_features": 10}), node("out", "Output")],
        [edge("in", "lin"), edge("lin", "out")],
    )
    assert "def train(model, X, y" in generate_training(g)
