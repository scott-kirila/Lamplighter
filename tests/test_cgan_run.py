"""End-to-end conditional-GAN run: the runner codegens two *conditional* models
(a class label, embedded, conditions both generator and discriminator), builds a
labeled (X, y) loader from the data-fed model minus its label port, trains with
the cgan recipe, and streams g_loss/d_loss — plus a resume round-trip. Exercised
through the real RunManager (synthetic data, injected namespace)."""
import torch

from lamplighter.backend.runner import RunManager
from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project
from tests.helpers import edge, graph, node


def _cgan_project(epochs=3):
    # Conditional generator: noise + label(→Embedding) → Concat → Linear → image.
    gen = graph(
        [node("noise", "Input", {"shape": "1, 100"}, y=0.0),
         node("glabel", "Input", {"shape": "1", "dtype": "long"}, y=1.0),
         node("gemb", "Embedding", {"num_embeddings": 10, "embedding_dim": 4}),
         node("gcat", "Concat", {"dim": 1}), node("gl", "Linear", {"out_features": 8}),
         node("gout", "Output")],
        [edge("noise", "gcat", tgt_h="in0"), edge("glabel", "gemb"),
         edge("gemb", "gcat", tgt_h="in1"), edge("gcat", "gl"), edge("gl", "gout")],
    )
    # Conditional discriminator: image + label(→Embedding) → Concat → Linear → 1.
    disc = graph(
        [node("image", "Input", {"shape": "1, 8"}, y=0.0),
         node("dlabel", "Input", {"shape": "1", "dtype": "long"}, y=1.0),
         node("demb", "Embedding", {"num_embeddings": 10, "embedding_dim": 4}),
         node("dcat", "Concat", {"dim": 1}), node("dl", "Linear", {"out_features": 1}),
         node("dout", "Output")],
        [edge("image", "dcat", tgt_h="in0"), edge("dlabel", "demb"),
         edge("demb", "dcat", tgt_h="in1"), edge("dcat", "dl"), edge("dl", "dout")],
    )
    return Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        data_nodes=[
            DataNode(id="z", kind="noise", name="Noise", config={"dims": "100"}),
            DataNode(id="ds", kind="dataset", name="Data",
                     config={"source": "memory", "x_var": "X", "y_var": "Y", "batch_size": 8}),
        ],
        links=[
            ModelLink(id="l1", source_data="z", target_model="g", target_input="noise"),
            ModelLink(id="l2", source_data="ds", source_pin="y", target_model="g", target_input="glabel"),
            ModelLink(id="l3", source_data="ds", source_pin="x", target_model="d", target_input="image"),
            ModelLink(id="l4", source_data="ds", source_pin="y", target_model="d", target_input="dlabel"),
        ],
        training={
            "recipe": "cgan", "epochs": epochs, "device": "cpu", "seed": 0,
            "roles": {"generator": "g", "discriminator": "d"},
            "per_role": {"generator": {"lr": 0.01}, "discriminator": {"lr": 0.01}},
        },
    )


def _data(n=40):
    return {"X": torch.rand(n, 8), "Y": torch.randint(0, 10, (n,))}


def test_cgan_run_trains_both_conditional_models_and_streams_losses():
    torch.manual_seed(0)
    epochs_seen: list[dict] = []
    mgr = RunManager()
    err = mgr.start(_cgan_project(epochs=3), namespace=_data(), emit=lambda m: epochs_seen.append(m))
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error

    assert set(mgr.models) == {"generator", "discriminator"}
    assert len(mgr.history["g_loss"]) == 3 and len(mgr.history["d_loss"]) == 3
    epoch_events = [m for m in epochs_seen if m["type"] == "run_epoch"]
    assert len(epoch_events) == 3
    assert set(epoch_events[-1]["metrics"]) == {"g_loss", "d_loss"}

    # The restored conditional generator runs on (noise, label).
    with torch.no_grad():
        out = mgr.models["generator"](torch.randn(2, 100), torch.randint(0, 10, (2,)))
    assert tuple(out.shape) == (2, 8)


def test_cgan_run_accepts_ui_style_per_input_picks():
    # The app stores picks in x_vars keyed by Input node id whenever the model
    # shows several inputs — even though the *loader* graph (discriminator minus
    # its label port) has only one left. Pre-flight must fall back to x_vars
    # when x_var is empty, or the Run button's happy path 400s (live-smoke find).
    project = _cgan_project(epochs=2)
    ds = next(dn for dn in project.data_nodes if dn.kind == "dataset")
    ds.config = {"source": "memory", "x_vars": {"image": "X", "dlabel": "Y"},
                 "y_var": "Y", "batch_size": 8}
    mgr = RunManager()
    err = mgr.start(project, namespace=_data(), emit=lambda m: None)
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    assert len(mgr.history["g_loss"]) == 2


def test_resume_a_cgan_continues_both_models():
    from lamplighter.backend import checkpoints

    checkpoints.clear()
    torch.manual_seed(0)
    ns = _data()
    mgr = RunManager()
    assert mgr.start(_cgan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    checkpoints.save("cgan-run", mgr)

    err = mgr.resume("cgan-run", checkpoints.load("cgan-run"), epochs=4, namespace=ns, emit=lambda m: None)
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    assert set(mgr.models) == {"generator", "discriminator"}
    assert len(mgr.history["g_loss"]) == 4 and len(mgr.history["d_loss"]) == 4
    checkpoints.clear()
