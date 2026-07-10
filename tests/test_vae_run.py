"""End-to-end VAE run: the runner codegens an encoder with two *named* outputs
(mu/logvar — read by name, so canvas order can't matter) and a decoder, trains
them jointly with the vae recipe (reparameterize + recon + beta·KL, one
optimizer), streams recon_loss/kl_loss, and resumes. Exercised through the real
RunManager (synthetic data, injected namespace)."""
import pytest
import torch

from backend.recipes import RECIPES
from backend.runner import RunManager
from backend.schema import DataNode, Graph, ModelDef, ModelLink, Project
from tests.helpers import edge, graph, node


def _encoder():
    # in(8) → shared Linear → ReLU → two heads: mu(3) and logvar(3), as named
    # Outputs. logvar placed ABOVE mu on the canvas, to prove name-not-order.
    g = graph(
        [node("in", "Input", {"shape": "8, 8"}),
         node("l", "Linear", {"out_features": 16}), node("r", "ReLU"),
         node("lmu", "Linear", {"out_features": 3}, y=200.0),
         node("llv", "Linear", {"out_features": 3}, y=0.0),
         node("omu", "Output", {"name": "mu"}, y=200.0),
         node("olv", "Output", {"name": "logvar"}, y=0.0)],
        [edge("in", "l"), edge("l", "r"), edge("r", "lmu"), edge("r", "llv"),
         edge("lmu", "omu"), edge("llv", "olv")],
    )
    return Graph(nodes=g.nodes, edges=g.edges)


def _decoder():
    g = graph(
        [node("in", "Input", {"shape": "8, 3"}),
         node("l", "Linear", {"out_features": 8}), node("s", "Sigmoid"),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "s"), edge("s", "out")],
    )
    return Graph(nodes=g.nodes, edges=g.edges)


def _vae_project(epochs=3, **training):
    return Project(
        models=[
            ModelDef(id="e", name="Encoder", graph=_encoder()),
            ModelDef(id="d", name="Decoder", graph=_decoder()),
        ],
        data_nodes=[DataNode(id="ds", kind="dataset", name="Data",
                             config={"source": "memory", "x_var": "X", "batch_size": 8})],
        links=[ModelLink(id="L", source_data="ds", target_model="e")],
        training={
            "recipe": "vae", "epochs": epochs, "device": "cpu", "seed": 0, "lr": 0.01,
            "roles": {"encoder": "e", "decoder": "d"},
            **training,
        },
    )


def test_vae_generate_emits_the_reparameterized_loop():
    src = RECIPES["vae"].generate(_vae_project(epochs=3, beta=0.5))
    assert "mu, logvar = enc.mu, enc.logvar" in src  # by name, not position
    assert "z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)" in src
    assert 'F.binary_cross_entropy(x_hat, real, reduction="sum")' in src
    assert "loss = recon_loss + 0.5 * kl_loss" in src  # beta baked in

    mse = RECIPES["vae"].generate(_vae_project(epochs=2, recon="mse"))
    assert 'F.mse_loss(x_hat, real, reduction="sum")' in mse


def test_vae_requires_named_encoder_outputs():
    project = _vae_project()
    for n in project.models[0].graph.nodes:
        if n.type == "Output":
            n.params["name"] = ""  # strip the names
    with pytest.raises(ValueError, match="named 'mu' and 'logvar'"):
        RECIPES["vae"].generate(project)
    # And the runner surfaces it as a clean start error, never an exec crash.
    err = RunManager().start(project, namespace={"X": torch.rand(24, 8)}, emit=lambda m: None)
    assert err is not None and "mu" in err


def test_vae_run_trains_jointly_and_streams_losses():
    torch.manual_seed(0)
    ns = {"X": torch.rand(40, 8)}  # [0, 1] for BCE reconstruction

    events: list[dict] = []
    mgr = RunManager()
    err = mgr.start(_vae_project(epochs=3), namespace=ns, emit=events.append)
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error

    assert set(mgr.models) == {"encoder", "decoder"}
    assert len(mgr.history["recon_loss"]) == 3 and len(mgr.history["kl_loss"]) == 3
    epoch_events = [m for m in events if m["type"] == "run_epoch"]
    assert set(epoch_events[-1]["metrics"]) == {"recon_loss", "kl_loss"}

    # The trained pair round-trips: encode by name, decode the sampled z.
    with torch.no_grad():
        enc = mgr.models["encoder"](torch.rand(4, 8))
        out = mgr.models["decoder"](enc.mu)
    assert tuple(out.shape) == (4, 8)
    assert out.min() >= 0 and out.max() <= 1  # Sigmoid output — BCE-valid


def test_resume_a_vae_continues_both_models():
    from backend import checkpoints

    checkpoints.clear()
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_vae_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    checkpoints.save("vae-run", mgr)

    err = mgr.resume("vae-run", checkpoints.load("vae-run"), epochs=4, namespace=ns, emit=lambda m: None)
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    assert len(mgr.history["recon_loss"]) == 4 and len(mgr.history["kl_loss"]) == 4
    checkpoints.clear()
