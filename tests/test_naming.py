"""Explicit Input/Output naming: a node's `name` param sets its forward()
argument name, and a multi-output model with any named Output returns a
namedtuple (so callers can unpack it *and* read fields by name)."""
import torch

from lamplighter.backend.codegen import generate_module
from lamplighter.backend.inference import graph_issues
from tests.helpers import edge, graph, node


def _two_output_graph(name1="logits", name2="aux"):
    # Input -> l1 -> Output(name1); l1 -> l2 -> Output(name2), stacked so o1 is
    # the top (first) return field.
    return graph(
        [
            node("in", "Input", {"shape": "4, 64"}),
            node("l1", "Linear", {"out_features": 10}),
            node("l2", "Linear", {"out_features": 5}),
            node("o1", "Output", {"name": name1}, y=0),
            node("o2", "Output", {"name": name2}, y=100),
        ],
        [edge("in", "l1"), edge("l1", "o1"), edge("l1", "l2"), edge("l2", "o2")],
    )


# --- Input naming ---------------------------------------------------------

def test_named_single_input():
    g = graph(
        [node("in", "Input", {"shape": "4, 64", "name": "image"}),
         node("l", "Linear", {"out_features": 10}), node("o", "Output")],
        [edge("in", "l"), edge("l", "o")],
    )
    assert "def forward(self, image):" in generate_module(g)


def test_named_multi_input_uses_names():
    g = graph(
        [
            node("a", "Input", {"shape": "4, 8", "name": "image"}, y=0),
            node("b", "Input", {"shape": "4, 8", "name": "mask"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("lin", "Linear", {"out_features": 10}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"),
         edge("cat", "lin"), edge("lin", "out")],
    )
    assert "def forward(self, image, mask):" in generate_module(g)


def test_blank_name_falls_back_to_auto():
    g = graph(
        [node("in", "Input", {"shape": "4, 64", "name": ""}),
         node("l", "Linear", {"out_features": 10}), node("o", "Output")],
        [edge("in", "l"), edge("l", "o")],
    )
    assert "def forward(self, x):" in generate_module(g)  # unchanged from unnamed


# --- Output naming (namedtuple) -------------------------------------------

def test_named_outputs_return_namedtuple():
    code = generate_module(_two_output_graph())
    assert 'ModelOutput = namedtuple("ModelOutput", [\'logits\', \'aux\'])' in code
    assert "return ModelOutput(logits=" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    out = ns["GeneratedModel"]().eval()(torch.randn(4, 64))
    # Both tuple-unpacking and attribute access work.
    logits, aux = out
    assert list(out.logits.shape) == [4, 10] and list(out.aux.shape) == [4, 5]
    assert list(logits.shape) == [4, 10] and list(aux.shape) == [4, 5]


def test_partially_named_outputs_autofill():
    # Only o1 named -> still a namedtuple; the blank field auto-names out1.
    code = generate_module(_two_output_graph(name1="logits", name2=""))
    assert 'namedtuple("ModelOutput", [\'logits\', \'out1\'])' in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    out = ns["GeneratedModel"]().eval()(torch.randn(4, 64))
    assert list(out.logits.shape) == [4, 10] and list(out.out1.shape) == [4, 5]


def test_single_named_output_stays_bare():
    # A name on a lone Output is ignored — you don't wrap a single return.
    g = graph(
        [node("in", "Input", {"shape": "4, 64"}), node("l", "Linear", {"out_features": 10}),
         node("o", "Output", {"name": "logits"})],
        [edge("in", "l"), edge("l", "o")],
    )
    code = generate_module(g)
    assert code.rstrip().endswith("return t0")
    assert "namedtuple" not in code


def test_unnamed_multi_output_is_plain_tuple():
    code = generate_module(_two_output_graph(name1="", name2=""))
    assert "namedtuple" not in code
    assert code.rstrip().endswith("return t0, t1")


# --- name validation ------------------------------------------------------

def test_invalid_identifier_flagged():
    g = graph([node("in", "Input", {"shape": "4, 64", "name": "my input"}), node("o", "Output")], [])
    assert "Input name 'my input' is not a valid identifier." in graph_issues(g)


def test_keyword_name_flagged():
    g = graph([node("in", "Input", {"shape": "4, 64", "name": "class"}), node("o", "Output")], [])
    assert "Input name 'class' is not a valid identifier." in graph_issues(g)


def test_duplicate_input_names_flagged():
    g = graph(
        [node("a", "Input", {"shape": "4, 8", "name": "x"}),
         node("b", "Input", {"shape": "4, 8", "name": "x"}), node("o", "Output")],
        [],
    )
    assert "Duplicate Input name 'x'." in graph_issues(g)


def test_leading_underscore_output_flagged():
    g = graph([node("in", "Input", {"shape": "4, 64"}), node("o", "Output", {"name": "_hidden"})], [])
    assert "Output name '_hidden' can't start with an underscore." in graph_issues(g)


def test_valid_names_produce_no_issues():
    g = graph(
        [node("in", "Input", {"shape": "4, 64", "name": "image"}),
         node("o", "Output", {"name": "logits"})],
        [edge("in", "o")],
    )
    assert graph_issues(g) == []
