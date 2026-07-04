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


def test_gan_checkpoint_is_v3_with_per_role_state_dicts():
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_gan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error

    ckpt = mgr.checkpoint()
    assert "state_dict" not in ckpt  # not the v2 shape
    assert set(ckpt["state_dicts"]) == {"generator", "discriminator"}
    assert ckpt["best_state_dict"] is None  # no best without validation
    assert ckpt["epoch"] == 2
    # The snapshot records the whole project + per-role sources (self-contained).
    assert "project" in ckpt["snapshot"]
    assert set(ckpt["snapshot"]["sources"]["models"]) == {"generator", "discriminator"}


def test_restore_a_gan_repopulates_both_models():
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_gan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    ckpt = mgr.checkpoint()

    fresh = RunManager()
    assert fresh.restore(ckpt) is None
    assert fresh.state == "done"
    assert set(fresh.models) == {"generator", "discriminator"}
    with torch.no_grad():
        out = fresh.models["generator"](torch.randn(3, 16))
    assert tuple(out.shape) == (3, 8)  # the restored generator runs


def test_resume_a_gan_continues_both_models():
    from backend import checkpoints

    checkpoints.clear()
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_gan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    checkpoints.save("gan-run", mgr)  # v3, weights cloned

    err = mgr.resume("gan-run", checkpoints.load("gan-run"), epochs=4, namespace=ns, emit=lambda m: None)
    assert err is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    assert set(mgr.models) == {"generator", "discriminator"}
    # 2 stored epochs + 2 resumed = one continuous 4-epoch curve.
    assert len(mgr.history["g_loss"]) == 4 and len(mgr.history["d_loss"]) == 4
    checkpoints.clear()


def test_gan_autosave_rolls_a_v3_entry():
    from backend import checkpoints

    checkpoints.clear()
    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    project = _gan_project(epochs=4)
    project.training = {**project.training, "autosave_every": 2}
    mgr = RunManager()
    assert mgr.start(project, namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)

    entry = checkpoints.load("autosave")
    assert set(entry["state_dicts"]) == {"generator", "discriminator"}
    checkpoints.clear()


def test_load_checkpoint_picks_a_gan_model(tmp_path):
    import lamplighter

    torch.manual_seed(0)
    ns = {"X": torch.rand(24, 8)}
    mgr = RunManager()
    assert mgr.start(_gan_project(epochs=2), namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    path = tmp_path / "gan.pt"
    torch.save(mgr.checkpoint(), path)

    # Ambiguous without a role.
    with pytest.raises(lamplighter.LamplighterError, match="several models"):
        lamplighter.load_checkpoint(str(path))
    generator, snapshot = lamplighter.load_checkpoint(str(path), model="generator")
    with torch.no_grad():
        assert tuple(generator(torch.randn(2, 16)).shape) == (2, 8)
    assert "project" in snapshot


def test_unassigned_gan_roles_are_rejected():
    project = _gan_project()
    project.training = {**project.training, "roles": {}}  # no assignment
    mgr = RunManager()
    error = mgr.start(project, namespace={"X": torch.rand(8, 8)}, emit=lambda m: None)
    assert error is not None and "generator" in error
