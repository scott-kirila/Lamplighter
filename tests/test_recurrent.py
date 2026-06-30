"""Multi-output (recurrent) layers: per-pin shape inference, codegen that only
materializes wired output pins, and a real forward pass."""
import torch

from backend.codegen import generate_module
from backend.inference import infer_shapes
from tests.helpers import edge, graph, node, output_id


def test_lstm_per_pin_shapes():
    # LSTM returns (output, (h_n, c_n)); each pin gets its own inferred shape.
    g = graph(
        [
            node("in", "Input", {"shape": "8, 5, 16"}),  # (batch, seq, feat), batch_first
            node("lstm", "LSTM", {"hidden_size": 32, "batch_first": True}),
            node("out", "Output"),
        ],
        [edge("in", "lstm"), edge("lstm", "out", src_h="output")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert shapes[("lstm", "output")] == [8, 5, 32]  # (batch, seq, hidden)
    assert shapes[("lstm", "h_n")] == [1, 8, 32]      # (layers, batch, hidden)
    assert shapes[("lstm", "c_n")] == [1, 8, 32]


def test_codegen_extracts_only_wired_pins():
    # Only `output` is wired downstream -> no extraction for h_n / c_n.
    g = graph(
        [
            node("in", "Input", {"shape": "8, 5, 16"}),
            node("lstm", "LSTM", {"hidden_size": 32, "batch_first": True}),
            node("flat", "Flatten"),
            node("lin", "Linear", {"out_features": 4}),
            node("out", "Output"),
        ],
        [edge("in", "lstm"), edge("lstm", "flat", src_h="output"), edge("flat", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)
    assert "t0 = self.layer_0(x)" in code  # the (output, (h_n, c_n)) tuple
    assert "t1 = t0[0]" in code            # output extracted
    assert "t0[1]" not in code             # hidden states never touched


def test_codegen_extracts_hidden_when_wired():
    # Wiring h_n downstream emits its indexed extraction; `output` stays unused.
    g = graph(
        [
            node("in", "Input", {"shape": "8, 5, 16"}),
            node("lstm", "LSTM", {"hidden_size": 32, "batch_first": True}),
            node("flat", "Flatten"),
            node("lin", "Linear", {"out_features": 4}),
            node("out", "Output"),
        ],
        [edge("in", "lstm"), edge("lstm", "flat", src_h="h_n"), edge("flat", "lin"), edge("lin", "out")],
    )
    code = generate_module(g)
    assert "t0[1][0]" in code   # h_n = result[1][0]
    assert "t0[0]" not in code  # output not wired, not extracted


def test_lstm_model_runs():
    g = graph(
        [
            node("in", "Input", {"shape": "8, 5, 16"}),
            node("lstm", "LSTM", {"hidden_size": 32, "batch_first": True}),
            node("flat", "Flatten"),
            node("lin", "Linear", {"out_features": 4}),
            node("out", "Output"),
        ],
        [edge("in", "lstm"), edge("lstm", "flat", src_h="output"), edge("flat", "lin"), edge("lin", "out")],
    )
    shapes, errors = infer_shapes(g)
    assert errors == {}
    code = generate_module(g)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    model = ns["GeneratedModel"]().eval()
    out = model(torch.randn(8, 5, 16))
    assert list(out.shape) == shapes[(output_id(g), "output")]
