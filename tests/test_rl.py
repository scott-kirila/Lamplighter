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
