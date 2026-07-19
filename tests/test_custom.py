"""The Custom node — the palette escape hatch.

A notebook-defined nn.Module class, registered via sess.modules(Name=Class), is
picked on the node with literal init args; codegen splices its source into the
generated module, so exports and checkpoints stay self-contained (the defining
property — everything here rebuilds WITHOUT the registry/session).
"""
import pytest
import torch
import torch.nn as nn

from lamplighter.backend import datastore
from lamplighter.backend.codegen import exec_generated, generate_module
from lamplighter.backend.inference import infer_shapes, primary_shapes
from lamplighter.backend.registry import parse_literal_args
from tests.helpers import edge, graph, node, single_model_project


class GatedBlock(nn.Module):
    """A gated residual unit — the kind of block the palette will never have."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return x + self.drop(a * torch.sigmoid(b))


class MetaHostile(nn.Module):
    """forward calls .item(), which meta tensors can't do — exercises the
    real-CPU probe fallback."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        scale = float(x.detach().abs().sum())  # .item()-ish: breaks on meta
        return self.fc(x) * (1.0 if scale >= 0 else 0.0)


class LocalImport(nn.Module):
    """Imports beyond torch/nn go inside methods — the documented v1 rule."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        import math

        return x * math.sqrt(self.dim)


@pytest.fixture(autouse=True)
def _clean_modules():
    datastore.clear_modules()
    yield
    datastore.clear_modules()


def _custom_graph(cls="GatedBlock", args="64, dropout=0.1"):
    return graph(
        [node("in", "Input", {"shape": "8, 64"}),
         node("c", "Custom", {"cls": cls, "args": args}),
         node("l", "Linear", {"out_features": 10}), node("out", "Output")],
        [edge("in", "c"), edge("c", "l"), edge("l", "out")],
    )


def _model_from(src):
    found = None
    for v in exec_generated(src, "<test-custom>").values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            found = v  # last wins — the model class follows the spliced ones
    return found


# --- registration ---------------------------------------------------------------

def test_register_modules_rejects_non_classes():
    with pytest.raises(ValueError, match="nn.Module subclass"):
        datastore.register_modules(bad=GatedBlock(64))  # instance, not class
    with pytest.raises(ValueError, match="nn.Module subclass"):
        datastore.register_modules(worse=42)
    datastore.register_modules(GatedBlock=GatedBlock)
    assert datastore.module_summaries()[0]["name"] == "GatedBlock"
    assert "gated residual" in datastore.module_summaries()[0]["doc"].lower()


# --- literal args ----------------------------------------------------------------

def test_init_args_are_literals_only():
    assert parse_literal_args("64, dropout=0.1") == ([64], {"dropout": 0.1})
    assert parse_literal_args("(3, 3), padding='same'") == ([(3, 3)], {"padding": "same"})
    assert parse_literal_args("") == ([], {})
    for evil in ("d // 2", "__import__('os')", "64; import os", "**{'a': 1}"):
        with pytest.raises(ValueError, match="literals"):
            parse_literal_args(evil)


# --- inference -------------------------------------------------------------------

def test_custom_infers_through_the_real_class():
    datastore.register_modules(GatedBlock=GatedBlock)
    g = _custom_graph()
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["c"] == [8, 64]  # residual: shape-preserving


def test_custom_param_counts_ride_inference():
    datastore.register_modules(GatedBlock=GatedBlock)
    counts: dict = {}
    infer_shapes(_custom_graph(), param_counts=counts)
    assert counts["c"]["count"] == 64 * 128 + 128  # fc: Linear(64 -> 128)


def test_meta_hostile_class_falls_back_to_a_cpu_probe():
    datastore.register_modules(MetaHostile=MetaHostile)
    g = _custom_graph(cls="MetaHostile", args="64")
    shapes, errors = infer_shapes(g)
    assert errors == {}
    assert primary_shapes(g, shapes)["c"] == [8, 64]


def test_unregistered_and_blank_names_error_clearly():
    g = _custom_graph(cls="Nope", args="")
    _, errors = infer_shapes(g)
    assert "not registered" in errors["c"] and "sess.modules(Nope=...)" in errors["c"]

    _, errors = infer_shapes(_custom_graph(cls="", args=""))
    assert "pick a registered module" in errors["c"]


def test_expression_args_surface_as_a_node_error():
    datastore.register_modules(GatedBlock=GatedBlock)
    _, errors = infer_shapes(_custom_graph(args="64 // 2"))
    assert "literals" in errors["c"]


# --- codegen: the splice ----------------------------------------------------------

def test_generated_module_is_self_contained():
    datastore.register_modules(GatedBlock=GatedBlock)
    src = generate_module(_custom_graph())
    # The user's class is spliced above the model, instantiated with the args.
    assert "class GatedBlock(nn.Module):" in src
    assert "self.layer_0 = GatedBlock(64, dropout=0.1)" in src
    assert src.index("class GatedBlock") < src.index("class GeneratedModel")

    # The defining property: rebuild + run with the registry GONE.
    datastore.clear_modules()
    model = _model_from(src)()
    assert model.__class__.__name__ == "GeneratedModel"  # last-wins finder
    assert tuple(model(torch.randn(8, 64)).shape) == (8, 10)


def test_two_nodes_one_class_splice_once():
    datastore.register_modules(GatedBlock=GatedBlock)
    g = graph(
        [node("in", "Input", {"shape": "8, 64"}),
         node("c1", "Custom", {"cls": "GatedBlock", "args": "64"}),
         node("c2", "Custom", {"cls": "GatedBlock", "args": "64"}),
         node("out", "Output")],
        [edge("in", "c1"), edge("c1", "c2"), edge("c2", "out")],
    )
    src = generate_module(g)
    assert src.count("class GatedBlock(nn.Module):") == 1  # deduped
    assert "self.layer_0 = GatedBlock(64)" in src and "self.layer_1 = GatedBlock(64)" in src


def test_method_local_imports_survive_the_splice():
    datastore.register_modules(LocalImport=LocalImport)
    src = generate_module(_custom_graph(cls="LocalImport", args="64"))
    datastore.clear_modules()
    out = _model_from(src)()(torch.randn(8, 64))
    assert tuple(out.shape) == (8, 10)


# --- the full loop: train + checkpoint without the session -------------------------

def test_custom_model_trains_and_its_checkpoint_rebuilds_without_the_registry(tmp_path):
    import lamplighter
    from lamplighter.backend.runner import RunManager

    datastore.register_modules(GatedBlock=GatedBlock)
    project = single_model_project(
        _custom_graph(),
        training={"epochs": 2, "device": "cpu", "seed": 0, "loss": "CrossEntropyLoss"},
        data={"source": "memory", "x_var": "X", "y_var": "y", "batch_size": 8},
    )
    ns = {"X": torch.randn(24, 64), "y": torch.randint(0, 10, (24,))}

    mgr = RunManager()
    assert mgr.start(project, namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error

    path = tmp_path / "custom.pt"
    torch.save(mgr.checkpoint(), path)
    datastore.clear_modules()  # a fresh kernel: nothing registered

    rebuilt, snapshot = lamplighter.load_checkpoint(str(path))
    assert "class GatedBlock" in snapshot["sources"]["models"]["model"]
    with torch.no_grad():
        assert tuple(rebuilt(torch.randn(4, 64)).shape) == (4, 10)
