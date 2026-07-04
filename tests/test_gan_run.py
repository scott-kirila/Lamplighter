"""End-to-end multi-model run: the runner builds two models, trains them with
the GAN recipe, and streams g_loss/d_loss — the core "GAN trains in-app"
capability, exercised through the real RunManager (synthetic data, injected
namespace, no globals)."""
import pytest
import torch

from backend.runner import RunManager
from backend.schema import Graph, ModelDef, Project
from tests.helpers import edge, graph, node


def _gan_project(epochs=3):
    gen = graph(
        [node("in", "Input", {"shape": "1, 16"}), node("l", "Linear", {"out_features": 8}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    disc = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 1}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        training={
            "recipe": "gan", "epochs": epochs, "device": "cpu", "seed": 0,
            "roles": {"generator": "g", "discriminator": "d"},
            "per_role": {"generator": {"lr": 0.01}, "discriminator": {"lr": 0.01}},
        },
        data={"source": "memory", "x_var": "X", "batch_size": 8},
    )


def test_gan_run_trains_both_models_and_streams_losses():
    torch.manual_seed(0)
    ns = {"X": torch.rand(40, 8)}  # "real" 8-dim data (the discriminator's input)

    epochs_seen: list[dict] = []
    mgr = RunManager()
    err = mgr.start(_gan_project(epochs=3), namespace=ns, emit=lambda m: epochs_seen.append(m))
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error

    # Both models came back, keyed by role.
    assert set(mgr.models) == {"generator", "discriminator"}
    assert mgr.model is None  # no single-model convenience for a multi-model run

    # g_loss/d_loss streamed for each epoch (generic metric plumbing, no GAN
    # branch in the runner).
    assert len(mgr.history["g_loss"]) == 3 and len(mgr.history["d_loss"]) == 3
    epoch_events = [m for m in epochs_seen if m["type"] == "run_epoch"]
    assert len(epoch_events) == 3
    assert set(epoch_events[-1]["metrics"]) == {"g_loss", "d_loss"}

    # Both models' parameters actually moved.
    assert all(p.abs().sum().item() > 0 for p in mgr.models["generator"].parameters())


def test_gan_run_refuses_checkpoint_for_now():
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_gan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    with pytest.raises(ValueError, match="multi-model"):
        mgr.checkpoint()


def test_unassigned_gan_roles_are_rejected():
    project = _gan_project()
    project.training = {**project.training, "roles": {}}  # no assignment
    mgr = RunManager()
    error = mgr.start(project, namespace={"X": torch.rand(8, 8)}, emit=lambda m: None)
    assert error is not None and "generator" in error
