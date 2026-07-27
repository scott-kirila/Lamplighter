"""The nn.Module importer — fx trace to a canvas graph, faithfully or not at all.

The load-bearing claim is round-trip fidelity: an imported model, run through the
UNCHANGED codegen and seeded positionally, must produce numerically identical
output. Everything else here defends that claim's edges — the gate that keeps a
wrong picture off the canvas, and the traps that silently corrupted the graph in
the prototype.
"""
import pytest
import torch
import torch.nn as nn

from lamplighter.backend.codegen import exec_generated, generate_module
from lamplighter.backend.importer import ImportError_, trace


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """The install/run tests mutate state._current and the datastore's import
    registry, both process-global. Snapshot and restore so a module that
    installs a model doesn't hand it to whatever test file runs next (the
    conftest tripwire that caught this exists for exactly that reason)."""
    from lamplighter.backend import datastore, state
    prior_project = state.get_project()
    prior_imports = dict(datastore._imports)
    yield
    state._current = prior_project
    datastore._imports.clear()
    datastore._imports.update(prior_imports)


def _roundtrip_maxdiff(model, shape):
    """Import, generate, seed with the original weights positionally, and return
    the max elementwise output difference — the whole keystone in one number."""
    model = model.eval()
    result = trace(model, shape)
    assert result["opaque_count"] == 0, result["findings"]
    src = generate_module(result["graph"], class_name="Imported")
    gen = exec_generated(src, "<test-import>")["Imported"]()
    gen_keys = list(gen.state_dict().keys())
    assert len(gen_keys) == len(result["state_keys"]), "state_dict key count must match"
    # Graph-ordered values, not registration order — the whole point of the fix.
    gen.load_state_dict(dict(zip(gen_keys, result["state_values"])))
    gen.eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        return (model(x) - gen(x)).abs().max().item()


# --- the keystone: numerically identical round-trip -------------------------

@pytest.mark.parametrize(
    "name,make,shape",
    [
        ("sequential_mlp",
         lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)),
         (1, 1, 28, 28)),
        ("conv_stack",
         lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)),
         (1, 3, 16, 16)),
        ("depthwise_separable",
         lambda: nn.Sequential(nn.Conv2d(8, 8, 3, groups=8, bias=False, padding=1),
                               nn.Conv2d(8, 16, 1), nn.ReLU()),
         (1, 8, 12, 12)),
    ],
)
def test_roundtrip_is_numerically_identical(name, make, shape):
    assert _roundtrip_maxdiff(make(), shape) < 1e-5


def test_resnet18_roundtrips(recwarn):
    """The headline case, kept separate because it constructs a torchvision
    model. 71 nodes, 122 state_dict keys, exact."""
    torchvision = pytest.importorskip("torchvision.models")
    assert _roundtrip_maxdiff(torchvision.resnet18(), (1, 3, 224, 224)) < 1e-5


# --- the importer must not corrupt the caller's model -----------------------

def test_trace_leaves_the_source_model_on_its_device():
    """ShapeProp moves a COPY to meta — a regression guard for the bug where
    gm.to('meta') stripped the weights off the shared submodules, so the very
    state_dict we then read for transfer was empty."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    before = model[0].weight.clone()
    trace(model, (1, 4))
    assert model[0].weight.device.type != "meta"
    assert torch.equal(model[0].weight, before)


# --- the fidelity gate: refuse rather than draw a wrong picture -------------

def test_a_param_the_node_cannot_carry_forces_opaque():
    """The gate's whole job: a value the registry node can't express becomes an
    Opaque hole, never a confidently-wrong node. GELU's `approximate='tanh'`
    isn't a param on the GELU node, so it must go Opaque — where plain GELU
    imports clean."""
    plain = trace(nn.Sequential(nn.Linear(4, 4), nn.GELU()).eval(), (1, 4))
    assert plain["opaque_count"] == 0

    tanh = trace(nn.Sequential(nn.Linear(4, 4), nn.GELU(approximate="tanh")).eval(), (1, 4))
    assert tanh["opaque_count"] == 1
    (finding,) = [f for f in tanh["findings"] if f["kind"] == "opaque"]
    assert "GELU" in finding["label"]


def test_an_unmapped_op_becomes_opaque_not_a_guess():
    class UsesUnknownOp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return torch.sigmoid(self.fc(x)) * 2.0  # `mul` has no registry node

        # `torch.sigmoid` maps to nothing in _FUNCS either — but sigmoid the
        # MODULE does; here it's the functional call, deliberately unmapped.

    result = trace(UsesUnknownOp().eval(), (1, 4))
    assert result["opaque_count"] >= 1


def test_codegen_refuses_a_graph_with_opaque_nodes_by_name():
    """An Opaque node generates no code; codegen must refuse the whole graph
    with a message naming the layer, not emit source that drops it."""
    result = trace(nn.Sequential(nn.Linear(4, 4), nn.GELU(approximate="tanh")).eval(), (1, 4))
    with pytest.raises(ValueError, match="GELU"):
        generate_module(result["graph"])


def test_transformer_is_refused_wholesale_not_shown_as_holes():
    torchvision = pytest.importorskip("torchvision.models")
    result = trace(torchvision.vit_b_16().eval(), (1, 3, 224, 224))
    assert result["refused"], "a mostly-plumbing model should be refused, not drawn"
    assert "can't represent" in result["refused_reason"]


# --- honest failure ---------------------------------------------------------

def test_recurrent_models_fail_with_torchs_own_message():
    with pytest.raises(ImportError_) as exc:
        trace(nn.LSTM(8, 16, batch_first=True), (1, 5, 8))
    assert "control flow" in str(exc.value)


def test_input_shape_needs_a_batch_dim():
    with pytest.raises(ImportError_, match="batch dim"):
        trace(nn.Linear(4, 4), (4,))


def test_tied_weights_are_refused_not_silently_untied():
    """A reused parametrized module would become two independent members and
    untie the shared weights — a silent correctness corruption, so it's an
    explicit refusal."""
    shared = nn.Linear(4, 4)

    class TiedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = shared

        def forward(self, x):
            return self.shared(self.shared(x))

    with pytest.raises(ImportError_, match="tied|used .* times"):
        trace(TiedNet().eval(), (1, 4))


# The test above asserts a general guarantee in its NAME but only ever covered
# one special case: a single module INSTANCE called twice. The other way to tie
# weights — two distinct modules sharing one Parameter object — is what every
# language model actually does, and it slipped straight through: the reuse
# guard counts fx targets, and `emb` and `head` are separate targets each
# called once. The import came out numerically exact (maxdiff 0.0), so the
# recorded "clean import ⟹ always exact" invariant read as holding, and the
# copies diverged to 7.8 after a single SGD step. A test whose name claims more
# than its body checks is worse than no test.

@pytest.mark.parametrize("build,shape", [
    # GPT-2 style: the output head shares the embedding matrix.
    (lambda: _tied_lm(), (2, 6)),
    # No Embedding involved — this one was silent under every prior guard.
    (lambda: _tied_linears(), (2, 8)),
])
def test_parameter_sharing_between_distinct_modules_is_refused(build, shape):
    with pytest.raises(ImportError_, match="ties weights"):
        trace(build().eval(), shape)


def _tied_lm():
    class TiedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(40, 8)
            self.head = nn.Linear(8, 40, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, x):
            return self.head(self.emb(x))

    return TiedLM()


def _tied_linears():
    class TiedLinears(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8, bias=False)
            self.b = nn.Linear(8, 8, bias=False)
            self.b.weight = self.a.weight

        def forward(self, x):
            return self.b(self.a(x))

    return TiedLinears()


def test_untied_models_are_not_caught_by_the_tie_guard():
    """The guard must not fire on ordinary models — two Linears with separately
    initialised weights of the same shape share no tensor."""
    plain = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    result = trace(plain.eval(), (2, 8))
    assert result["opaque_count"] == 0 and not result["refused"]


# --- the list-arg trap (torch.cat) ------------------------------------------

def test_concat_finds_its_inputs_through_the_list_arg():
    """torch.cat wraps its tensors in a list, so a naive Node-in-args scan finds
    zero sources and Concat reports no inputs. Both branches of a 2-way concat
    with different widths must be wired, in order."""
    class TwoBranchConcat(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 3)
            self.b = nn.Linear(4, 5)

        def forward(self, x):
            return torch.cat([self.a(x), self.b(x)], dim=1)  # -> width 8

    result = trace(TwoBranchConcat().eval(), (1, 4))
    assert result["opaque_count"] == 0
    concat = next(n for n in result["graph"].nodes if n.type == "Concat")
    incoming = [e for e in result["graph"].edges if e.target == concat.id]
    assert len(incoming) == 2, "both concat branches must be wired"
    # And it runs — a concat that dropped a branch would fail shape inference.
    assert _roundtrip_maxdiff(TwoBranchConcat(), (1, 4)) < 1e-5


# --- installing an import into a project and running it ----------------------

def _fresh_state():
    from lamplighter.backend import datastore, state
    state._current = None
    datastore._imports.clear()


def test_seed_from_weights_refuses_a_count_mismatch():
    from lamplighter.backend.importer import seed_from_weights

    model = nn.Linear(4, 4)  # 2 tensors (weight, bias)
    with pytest.raises(ImportError_, match="don't fit|key count"):
        seed_from_weights(model, [torch.zeros(4, 4)], ["only.one"])  # 1 value


def test_seed_from_weights_refuses_a_shape_mismatch():
    from lamplighter.backend.importer import seed_from_weights

    model = nn.Linear(4, 4)
    bad = [torch.zeros(4, 4), torch.zeros(99)]  # bias wrong shape
    with pytest.raises(ImportError_, match="refusing to mis-seed|expects"):
        seed_from_weights(model, bad, ["w", "b"])


def test_inspect_installs_a_runnable_model_and_stashes_its_weights():
    from lamplighter.backend import datastore, state
    from lamplighter.backend.import_install import inspect_model

    _fresh_state()
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).eval()
    report = inspect_model(model, torch.zeros(1, 1, 28, 28))

    assert report["installed"] and report["runnable"] and report["opaque"] == 0
    project = state.get_project()
    assert len(project.models) == 1
    md = project.models[0]
    assert md.imported is not None and md.imported.source == "Sequential"
    # The weights are in the kernel, keyed by model id — NOT in the graph/project.
    assert datastore.import_weights(md.id) is not None
    assert "imported" not in md.graph.model_dump()  # graph carries no weights


def test_a_refused_model_is_reported_but_not_installed():
    from lamplighter.backend import state
    from lamplighter.backend.import_install import inspect_model

    _fresh_state()
    torchvision = pytest.importorskip("torchvision.models")
    report = inspect_model(torchvision.vit_b_16().eval(), torch.zeros(1, 3, 224, 224))
    assert report["refused"] and not report["installed"]
    assert state.get_project() is None or not state.get_project().models


def test_running_an_import_starts_from_its_weights_not_a_fresh_init():
    """The property that makes import worth more than a picture: a run seeds the
    generated model with the ORIGINAL weights. Proven with lr=0 — one epoch that
    can't change a weight — so the trained state_dict must equal the imported
    one exactly, tensor for tensor."""
    from lamplighter.backend import state
    from lamplighter.backend.import_install import inspect_model
    from lamplighter.backend.runner import RunManager
    from lamplighter.backend.schema import DataNode, ModelLink

    _fresh_state()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 4)).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.7)  # recognizable, non-default values
    original = [v.clone() for v in model.state_dict().values()]

    report = inspect_model(model, torch.zeros(1, 1, 8, 8))
    project = state.get_project()
    mid = project.models[0].id
    project.training = {"device": "cpu", "loss": "CrossEntropyLoss", "epochs": 1,
                        "optimizer": "SGD", "lr": 0.0}  # lr 0 → weights frozen
    project.data_nodes = [DataNode(id="d", kind="dataset", name="D",
                                   config={"source": "memory", "x_var": "X", "y_var": "y"})]
    project.links = [ModelLink(id="l", source_data="d", target_model=mid)]
    state.set_project(project)

    mgr = RunManager()
    ns = {"X": torch.randn(32, 1, 8, 8), "y": torch.randint(0, 4, (32,))}
    assert mgr.start(project, namespace=ns, emit=lambda m: None) is None
    assert mgr.join(60) and mgr.state == "done", mgr.error

    trained = list(mgr.model.state_dict().values())
    assert len(trained) == len(original)
    for got, want in zip(trained, original):
        assert torch.equal(got, want), "the run didn't start from the imported weights"
    assert report["installed"]


def test_a_run_after_a_kernel_restart_says_the_weights_are_gone():
    """Imported weights live in the kernel; a restart clears them. The run must
    say so, not silently train a random init."""
    from lamplighter.backend import datastore, state
    from lamplighter.backend.import_install import inspect_model
    from lamplighter.backend.runner import RunManager
    from lamplighter.backend.schema import DataNode, ModelLink

    _fresh_state()
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 4)).eval()
    inspect_model(model, torch.zeros(1, 1, 8, 8))
    project = state.get_project()
    mid = project.models[0].id
    project.training = {"device": "cpu", "loss": "CrossEntropyLoss", "epochs": 1}
    project.data_nodes = [DataNode(id="d", kind="dataset", name="D",
                                   config={"source": "memory", "x_var": "X", "y_var": "y"})]
    project.links = [ModelLink(id="l", source_data="d", target_model=mid)]
    state.set_project(project)

    datastore._imports.clear()  # the "restart"
    mgr = RunManager()
    ns = {"X": torch.randn(8, 1, 8, 8), "y": torch.randint(0, 4, (8,))}
    err = mgr.start(project, namespace=ns, emit=lambda m: None)
    assert err is not None and "weights aren't in the kernel" in err


# --- the five silent-corruption defects the adversarial matrix found ---------
# Each is a case where trace() once returned opaque_count==0/refused==False (a
# claim of exact fidelity) but the graph was wrong. The invariant they enforce:
# a clean import (0 opaque, not refused) MUST round-trip exactly.

def _assert_clean_import_is_faithful(model, shape):
    """A clean import must be exact; a gap must show as opaque/refusal. Never a
    clean-looking import that diverges — that is the one forbidden state."""
    model = model.eval()
    r = trace(model, shape)
    if r["opaque_count"] or r["refused"]:
        return  # honest gap — acceptable
    diff = _roundtrip_maxdiff(model, shape)
    assert diff < 1e-5, f"clean import (0 opaque) but maxdiff {diff} — a WRONG PICTURE"


def test_cat_of_more_than_two_branches_goes_opaque_not_truncated():
    """#1: a 4-way cat was silently truncated to 2 branches (dropping googlenet's
    Inception convs) with opaque_count still 0."""
    class CatNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv2d(3, 4, 3, padding=1)
            s.c2 = nn.Conv2d(3, 4, 3, padding=1)
            s.pool = nn.MaxPool2d(3, 1, 1)
            s.act = nn.ReLU()

        def forward(s, x):
            return torch.cat([s.c1(x), s.c2(x), s.pool(x), s.act(x)], dim=1)

    r = trace(CatNet().eval(), (2, 3, 8, 8))
    assert r["opaque_count"] >= 1, "a 4-way cat must not be silently truncated"
    _assert_clean_import_is_faithful(CatNet(), (2, 3, 8, 8))


def test_mean_over_a_nondefault_dim_is_faithful_or_opaque():
    """#2: x.mean(dim=2) emitted the node default mean(dim=1) — same output
    shape, wrong values, 0 opaque. The worst kind: nothing but a value compare
    reveals it."""
    class MeanDim2(nn.Module):
        def __init__(s):
            super().__init__()
            s.lin = nn.Linear(8, 8)

        def forward(s, x):
            return s.lin(x).mean(dim=2)

    _assert_clean_import_is_faithful(MeanDim2(), (4, 8, 8))


def test_global_average_pool_over_a_dim_list_goes_opaque():
    """#2 (mnasnet's global-avg-pool): x.mean([2, 3]) can't be a single-dim Mean
    node — it must go Opaque, not silently reduce dim 1."""
    class GAP(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Conv2d(3, 6, 3, padding=1)

        def forward(s, x):
            return s.conv(x).mean([2, 3])

    r = trace(GAP().eval(), (2, 3, 8, 8))
    assert r["opaque_count"] >= 1
    _assert_clean_import_is_faithful(GAP(), (2, 3, 8, 8))


def test_flatten_with_a_bounded_end_dim_is_faithful_or_opaque():
    """#4: torch.flatten(x, 2) dropped start_dim and diverged in shape."""
    class FlattenMid(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Conv2d(3, 4, 3, padding=1)

        def forward(s, x):
            return torch.flatten(s.conv(x), 2)

    _assert_clean_import_is_faithful(FlattenMid(), (4, 3, 8, 8))


def test_submodules_applied_out_of_registration_order_seed_correctly():
    """#3: a ModuleList applied in reverse was seeded positionally against
    registration order — 0 opaque, guards passed, maxdiff ~1. Weights must
    follow the GRAPH's layer order."""
    class RevList(nn.Module):
        def __init__(s):
            super().__init__()
            s.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            # distinct weights so an order swap is detectable
            nn.init.constant_(s.layers[0].weight, 0.1)
            nn.init.constant_(s.layers[1].weight, 0.9)

        def forward(s, x):
            for layer in reversed(s.layers):
                x = layer(x)
            return x

    _assert_clean_import_is_faithful(RevList(), (2, 4))


def test_batchnorm_with_none_momentum_does_not_crash():
    """#5: momentum=None (a legal cumulative-average config) crashed trace() in
    _coerce's float(None). It must import (or Opaque), never traceback."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1),
                          nn.BatchNorm2d(4, momentum=None), nn.ReLU()).eval()
    r = trace(model, (2, 3, 8, 8))  # must not raise
    assert "graph" in r
    _assert_clean_import_is_faithful(model, (2, 3, 8, 8))


# --- and the models that exposed them, at the real scale ---------------------

def test_googlenet_does_not_import_as_a_wrong_picture():
    torchvision = pytest.importorskip("torchvision.models")
    _assert_clean_import_is_faithful(torchvision.googlenet(aux_logits=False), (1, 3, 224, 224))


def test_mnasnet_does_not_import_as_a_wrong_picture():
    torchvision = pytest.importorskip("torchvision.models")
    _assert_clean_import_is_faithful(torchvision.mnasnet1_0(), (1, 3, 224, 224))


# --- what the traced input actually is -------------------------------------------

class _TokenMLP(nn.Module):
    """A token-id model: the input indexes an Embedding, so it is long, not float."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, ids):
        return self.head(self.emb(ids).mean(dim=1))


def test_embedding_input_is_typed_long_not_float():
    """Every imported Input used to be stamped `float`.

    For a model whose first layer is an Embedding that is simply wrong, and it
    cost twice: the pre-flight panel reported the user's real token ids as
    "'X' is integer but the Input expects float" — a false error on exactly the
    models the importer was extended to cover — and the float probe raised
    inside ShapeProp, costing every Opaque node its recorded shape.
    """
    inp = next(n for n in trace(_TokenMLP(), (2, 7))["graph"].nodes if n.type == "Input")
    assert inp.params["dtype"] == "long"


def test_float_model_input_is_still_float():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)

        def forward(self, x):
            return self.a(x)

    inp = next(n for n in trace(Plain(), (2, 8))["graph"].nodes if n.type == "Input")
    assert inp.params["dtype"] == "float"


def test_observed_shapes_are_actually_recorded():
    """ShapeProp runs against a deepcopy, whose nodes are different OBJECTS from
    the ones fx_to_id was built against — so an identity lookup missed every
    time and this silently returned {} for every model ever imported. Opaque
    nodes then had no recorded shape, which is precisely what downstream
    inference needs to survive a hole in the graph."""
    result = trace(_TokenMLP(), (2, 7))
    assert result["observed_shapes"], "no shapes recorded at all"
    # The head's output shape is the one a class-range check would rely on.
    assert [2, 4] in result["observed_shapes"].values()


def test_opaque_nodes_carry_the_shape_inference_needs():
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.att = nn.MultiheadAttention(16, 2, batch_first=True)
            self.head = nn.Linear(16, 5)

        def forward(self, ids):
            h = self.emb(ids)
            a, _ = self.att(h, h, h)
            return self.head((h + a).mean(dim=1))

    graph = trace(TinyTransformer(), (2, 7))["graph"]
    opaque = [n for n in graph.nodes if n.type == "Opaque"]
    assert opaque, "expected the attention block to go Opaque"
    # Not every Opaque can have one (MultiheadAttention returns a tuple, so
    # there is no single shape to record) — but the tensor-valued ones must.
    assert any(n.params.get("out_shape") for n in opaque)
