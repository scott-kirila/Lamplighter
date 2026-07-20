"""RL Phase A: the env data seam + the REINFORCE recipe. An env recipe's data
source is a wired environment node — no loader path runs; the generated train()
creates the env itself, bakes the run's recorded seed (rollouts replay), samples
actions from Categorical over the policy's LOGITS, and streams per-episode
returns as step metrics. Convergence is the notebook's job — these tests pin
mechanics and determinism."""
import pytest

from lamplighter.backend.recipes import RECIPES, RL_ENVS
from lamplighter.backend.runner import RunManager
from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project
from tests.helpers import edge, graph, node
from tests.test_runner import JOIN_TIMEOUT


def _policy_graph():
    g = graph(
        [node("in", "Input", {"shape": "1, 4"}),
         node("l", "Linear", {"out_features": 2}),  # CartPole: 2 action logits
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return Graph(nodes=g.nodes, edges=g.edges)


def _rl_project(env_id="CartPole-v1", wire_env=True, **training):
    data_nodes = [DataNode(id="e", kind="env", name="Env", config={"env_id": env_id})]
    links = [ModelLink(id="L", source_data="e", target_model="policy")] if wire_env else []
    return Project(
        models=[ModelDef(id="policy", name="Policy", graph=_policy_graph())],
        data_nodes=data_nodes if wire_env else [],
        links=links,
        training={"recipe": "reinforce", "epochs": 2, "episodes_per_iter": 2,
                  "lr": 0.05, "seed": 0, **training},
    )


# --- generation --------------------------------------------------------------

def test_reinforce_generates_the_seeded_logits_first_loop():
    src = RECIPES["reinforce"].generate(_rl_project())
    assert "env = gym.make('CartPole-v1')" in src
    assert "Categorical(logits=policy(x))" in src  # logits-first — no softmax head
    assert "base_seed = 0" in src  # the recorded seed is IN the shown code
    assert "env.reset(seed=None if base_seed is None else base_seed + episode)" in src
    assert "g = r + 0.99 * g" in src  # returns-to-go with the default gamma
    assert "(returns - returns.mean()) / (returns.std() + 1e-8)" in src  # group baseline
    assert "entropy" in src and "0.01" in src  # the exploration bonus
    assert "make_dataloaders" not in src  # the env IS the data source


def test_reinforce_validates_the_env_wiring():
    with pytest.raises(ValueError, match="wire an environment"):
        RECIPES["reinforce"].generate(_rl_project(wire_env=False))
    with pytest.raises(ValueError, match="unknown environment 'Nope-v0'"):
        RECIPES["reinforce"].generate(_rl_project(env_id="Nope-v0"))
    assert "CartPole-v1" in RL_ENVS  # the curated list anchors the enum


# --- the run itself ----------------------------------------------------------

def test_reinforce_runs_records_and_streams_episode_returns():
    events: list[dict] = []
    mgr = RunManager()
    err = mgr.start(_rl_project(), namespace={}, emit=events.append)
    assert err is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error

    # The standard RL curve set, one point per iteration.
    assert set(mgr.history) == {"mean_return", "episode_len", "policy_loss", "entropy"}
    assert all(len(v) == 2 for v in mgr.history.values())
    assert all(v == v for v in mgr.history["mean_return"])  # finite

    # The policy trained (weights moved) and is the sole exposed model.
    assert set(mgr.models) == {"policy"}

    # Per-episode returns rode the step stream (throttled — at least the first),
    # each mapped onto the iteration axis for the chart.
    steps = [e for e in events if e.get("type") == "run_step"]
    assert steps and "episode_return" in steps[0]["metrics"]
    assert steps[0]["total"] == 4  # iterations × episodes sizes the axis up front
    assert "epoch_x" in steps[0]

    # The snapshot records the env verbatim (no dataset-form defaults) and the
    # trainer source shows the baked seed — the full reproducibility story.
    assert mgr.snapshot["data"] == {"env_id": "CartPole-v1"}
    assert "base_seed = 0" in mgr.snapshot["sources"]["trainer"]
    assert mgr.snapshot["sources"]["data"] is None


def test_reinforce_is_deterministic_under_a_pinned_seed():
    def run():
        mgr = RunManager()
        assert mgr.start(_rl_project(), namespace={}, emit=lambda m: None) is None
        assert mgr.join(JOIN_TIMEOUT)
        assert mgr.state == "done", mgr.error
        return mgr.history

    assert run() == run()  # seeded env resets + seeded torch sampling → identical


def test_reinforce_honors_the_cooperative_stop():
    mgr = RunManager()
    err = mgr.start(
        _rl_project(epochs=50),
        namespace={},
        emit=lambda m: mgr.stop() if m.get("type") == "run_epoch" else None,
    )
    assert err is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "stopped"
    assert len(mgr.history["mean_return"]) < 50


def test_reinforce_resumes_toward_a_higher_target():
    mgr = RunManager()
    assert mgr.start(_rl_project(), namespace={}, emit=lambda m: None) is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error

    err = mgr.resume("rl-run", mgr.checkpoint(), epochs=4, namespace={}, emit=lambda m: None)
    assert err is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error
    # One continuous curve: 2 stored iterations + 2 resumed.
    assert len(mgr.history["mean_return"]) == 4
    # The resumed trainer bakes the REMAINING count and its own fresh seed.
    assert "for iteration in range(2):" in mgr.snapshot["sources"]["trainer"]


# --- GRPO (clipped group-relative policy gradient) ---------------------------

def test_grpo_generates_the_clipped_group_relative_loop():
    src = RECIPES["grpo"].generate(_rl_project(recipe="grpo"))
    assert "Categorical(logits=policy(x" in src  # logits-first
    assert "(returns - returns.mean()) / (returns.std() + 1e-8)" in src  # group baseline
    assert "old_logp = torch.distributions.Categorical" in src  # frozen-policy ratio
    assert "ratio = torch.exp(dist.log_prob(act_batch) - old_logp)" in src
    assert "torch.clamp(ratio, 0.8, 1.2)" in src  # clip 0.2 → [0.8, 1.2]
    assert "torch.min(ratio * adv, clipped * adv)" in src  # the clipped surrogate
    assert "for _ in range(4):" in src  # default update_epochs reuse the rollout
    assert "value" not in src.lower().split("history")[0] or "value net" in src  # no critic


def test_grpo_runs_records_and_is_deterministic():
    def run():
        mgr = RunManager()
        assert mgr.start(_rl_project(recipe="grpo", epochs=2, episodes_per_iter=2, update_epochs=2),
                         namespace={}, emit=lambda m: None) is None
        assert mgr.join(JOIN_TIMEOUT)
        assert mgr.state == "done", mgr.error
        return mgr

    mgr = run()
    assert set(mgr.history) == {"mean_return", "episode_len", "policy_loss", "entropy"}
    assert all(len(v) == 2 for v in mgr.history.values())
    assert set(mgr.models) == {"policy"}
    # Seeded env resets + seeded sampling → bit-identical across runs.
    assert run().history == run().history


def test_grpo_recipe_shape():
    r = RECIPES["grpo"]
    assert [role.role for role in r.roles] == ["policy"]  # single role — no critic
    assert r.data == "env" and r.needs_targets is False and r.has_val is False
    assert {p.name for p in r.params} >= {"clip", "update_epochs", "gamma", "entropy_beta"}


# --- rollout replay (RL's preview) -------------------------------------------

def test_rollout_replays_the_policy_with_frames_and_probs():
    mgr = RunManager()
    assert mgr.start(_rl_project(), namespace={}, emit=lambda m: None) is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error

    r = mgr.rollout(max_steps=50)
    assert "error" not in r, r
    assert r["env_id"] == "CartPole-v1"
    assert 1 <= r["steps"] <= 50 and r["total_return"] > 0
    # A filmstrip of painted-ready frames: flat uint8 RGB, h·w·3 values each.
    assert len(r["frames"]) == len(r["probs"]) == len(r["actions"]) == len(r["returns"])
    f = r["frames"][0]
    assert f["h"] > 0 and f["w"] > 0 and len(f["data"]) == f["h"] * f["w"] * 3
    assert all(isinstance(v, int) for v in f["data"][:8])
    # Probabilities per step: one per action, summing to ~1 (display softmax).
    assert len(r["probs"][0]) == 2
    assert abs(sum(r["probs"][0]) - 1.0) < 0.01
    # The tally is cumulative and ends at the total.
    assert r["returns"][-1] == r["total_return"]

    # The stored-run flavor rebuilds from the checkpoint, kernel untouched.
    r2 = mgr.rollout_checkpoint(mgr.checkpoint(), max_steps=20)
    assert "error" not in r2 and r2["env_id"] == "CartPole-v1"


def test_rollout_episodes_are_indexed_and_reproducible():
    mgr = RunManager()
    assert mgr.start(_rl_project(), namespace={}, emit=lambda m: None) is None
    assert mgr.join(JOIN_TIMEOUT)
    assert mgr.state == "done", mgr.error

    # Episode 0 (the default) is the run's canonical replay — identical every time.
    r0a, r0b = mgr.rollout(max_steps=100), mgr.rollout(max_steps=100)
    assert (r0a["actions"], r0a["frames"]) == (r0b["actions"], r0b["frames"])

    # Episode k plays under seed+k: a genuinely different episode, itself replayable.
    r1a = mgr.rollout(max_steps=100, episode=1)
    r1b = mgr.rollout(max_steps=100, episode=1)
    assert (r1a["actions"], r1a["frames"]) == (r1b["actions"], r1b["frames"])
    assert (r1a["actions"], r1a["frames"]) != (r0a["actions"], r0a["frames"])

    # The stored-run flavor seeds the same way — flipping between the live and
    # the rebuilt policy shows the same episode k.
    rc = mgr.rollout_checkpoint(mgr.checkpoint(), max_steps=100, episode=1)
    assert rc["actions"] == r1a["actions"]


def test_rollout_refuses_a_non_rl_run():
    from tests.test_runner import _mlp_graph, _ns

    mgr = RunManager()
    assert mgr.start(_mlp_graph({"epochs": 1}), namespace=_ns(), emit=lambda m: None) is None
    assert mgr.join(JOIN_TIMEOUT)
    r = mgr.rollout()
    assert "wasn't an environment" in r["error"]


# --- env-aware shape evidence --------------------------------------------------

def test_env_node_output_shape_is_the_observation_space():
    from lamplighter.backend.inference import data_node_output_shape

    dn = DataNode(id="e", kind="env", name="Env", config={"env_id": "CartPole-v1"})
    assert data_node_output_shape(dn, {}) == [1, 4]  # CartPole observes 4 floats
    bad = DataNode(id="e", kind="env", name="Env", config={"env_id": "Nope-v0"})
    assert data_node_output_shape(bad, {}) is None  # uninspectable → no verdict


def test_diagnose_checks_obs_and_action_fit():
    from lamplighter.backend.diagnose import diagnose

    # The good project: obs (4) and 2 logits both match.
    rows = diagnose(_rl_project(), namespace={})
    assert any("observations (4) match the Input" in r["title"] for r in rows)
    assert any("2 action logits match CartPole-v1" in r["title"] for r in rows)

    # Wrong Input shape → the fix names the right one.
    p = _rl_project()
    p.models[0].graph.nodes[0].params["shape"] = "1, 6"
    rows = diagnose(p, namespace={})
    assert any(r["level"] == "error" and "observes (4)" in r["title"] for r in rows)

    # Wrong action head → the class-range-style error.
    p = _rl_project()
    p.models[0].graph.nodes[1].params["out_features"] = 3
    rows = diagnose(p, namespace={})
    assert any(
        r["level"] == "error" and "3 logits but CartPole-v1 has 2 actions" in r["title"]
        for r in rows
    )


def test_diagnose_flags_batch_hostile_policy_layers():
    from lamplighter.backend.diagnose import diagnose

    # RL steps the policy one observation at a time: BatchNorm errors outright
    # (train mode wants n>1) and Dropout resamples masks between forwards.
    g = graph(
        [node("in", "Input", {"shape": "1, 4"}),
         node("l1", "Linear", {"out_features": 8}),
         node("bn", "BatchNorm1d", {}),
         node("dp", "Dropout", {}),
         node("l2", "Linear", {"out_features": 2}),
         node("out", "Output")],
        [edge("in", "l1"), edge("l1", "bn"), edge("bn", "dp"), edge("dp", "l2"), edge("l2", "out")],
    )
    p = _rl_project()
    p.models[0].graph = Graph(nodes=g.nodes, edges=g.edges)

    rows = diagnose(p, namespace={})
    assert any(
        r["level"] == "error"
        and "BatchNorm1d can't train on single observations" in r["title"]
        and "LayerNorm" in r["detail"]
        for r in rows
    )
    assert any(
        r["level"] == "warn" and "Dropout resamples its mask" in r["title"] for r in rows
    )

    # The lint is env-recipe-scoped — the same layers are legitimate under a
    # batched (supervised) recipe.
    p.training = {**p.training, "recipe": "supervised"}
    rows = diagnose(p, namespace={})
    assert not any(
        "single observations" in r["title"] or "resamples its mask" in r["title"] for r in rows
    )


# --- guards ------------------------------------------------------------------

def test_missing_gymnasium_surfaces_the_install_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "gymnasium", None)
    mgr = RunManager()
    err = mgr.start(_rl_project(), namespace={}, emit=lambda m: None)
    assert err is not None and 'pip install "lamplighter[rl]"' in err


def test_unwired_env_fails_the_start_with_the_fix():
    mgr = RunManager()
    err = mgr.start(_rl_project(wire_env=False), namespace={}, emit=lambda m: None)
    assert err is not None and "wire an environment" in err


def test_diagnose_checks_the_env_wiring():
    from lamplighter.backend.diagnose import diagnose

    rows = diagnose(_rl_project(), namespace={})
    assert any(r["level"] == "ok" and "environment: CartPole-v1" in r["title"] for r in rows)

    rows = diagnose(_rl_project(wire_env=False), namespace={})
    assert any(r["level"] == "error" and "No environment wired" in r["title"] for r in rows)

    rows = diagnose(_rl_project(env_id="Nope-v0"), namespace={})
    assert any(r["level"] == "error" and "unknown environment" in r["title"] for r in rows)


def test_recipes_endpoint_declares_the_data_kind():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    with TestClient(app) as c:
        recipes = {r["name"]: r for r in c.get("/api/recipes").json()}
    assert recipes["reinforce"]["data"] == "env"
    assert recipes["reinforce"]["roles"] == [{"role": "policy", "label": "Policy"}]
    assert any(p["name"] == "episodes_per_iter" for p in recipes["reinforce"]["params"])
    assert recipes["supervised"]["data"] == "loader"  # the default, declared
    # Sweepable metrics are the curves each loop RECORDS — the Optimize select
    # must never offer a GAN val_loss its history won't contain.
    assert recipes["reinforce"]["metrics"] == ["mean_return"]
    assert recipes["grpo"]["metrics"] == ["mean_return"]
    assert recipes["supervised"]["metrics"] == ["val_loss", "train_loss"]
    assert recipes["gan"]["metrics"] == ["g_loss", "d_loss"]
    assert recipes["cgan"]["metrics"] == ["g_loss", "d_loss"]
    assert recipes["vae"]["metrics"] == ["recon_loss", "kl_loss"]
