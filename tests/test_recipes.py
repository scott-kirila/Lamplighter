"""The training recipe registry.

Pins the recipe abstraction that makes the GAN loop (Phase E) a data addition:
the supervised recipe must generate byte-identical source to today's
``generate_training`` (so the classic single-model flow is untouched), the
registry must be well-formed, and the runner/API must route through it.
"""
from backend.codegen import generate_training
from backend.recipes import DEFAULT_RECIPE, RECIPES, get_recipe
from backend.schema import project_from_graph
from tests.helpers import edge, graph, node


def _classifier(training=None, data=None):
    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("a", "Linear", {"out_features": 16}),
         node("r", "ReLU"), node("b", "Linear", {"out_features": 3}), node("out", "Output")],
        [edge("in", "a"), edge("a", "r"), edge("r", "b"), edge("b", "out")],
    )
    g.training = training or {}
    g.data = data or {}
    return g


def _multi_input():
    g = graph(
        [node("x0", "Input", {"shape": "1, 4"}), node("x1", "Input", {"shape": "1, 6"}),
         node("cat", "Concat", {"dim": 1}), node("l", "Linear", {"out_features": 2}), node("out", "Output")],
        [edge("x0", "cat", tgt_h="in0"), edge("x1", "cat", tgt_h="in1"), edge("cat", "l"), edge("l", "out")],
    )
    g.training = {"loss": "MSELoss", "metric": "none", "epochs": 7}
    g.data = {"val_split": 0.2}
    return g


# --- the golden byte-identical guarantee -----------------------------------

def test_supervised_is_byte_identical_to_generate_training():
    cases = [
        _classifier(),
        _classifier({"loss": "CrossEntropyLoss", "optimizer": "SGD", "lr": 0.05, "epochs": 20}),
        _classifier({"loss": "MSELoss", "metric": "none", "weight_decay": 1e-4}),
        _classifier({"metric": "accuracy"}, {"val_split": 0.25}),
        _multi_input(),
    ]
    gen = RECIPES["supervised"].generate
    for g in cases:
        assert gen(project_from_graph(g)) == generate_training(g)


# --- dispatch --------------------------------------------------------------

def test_get_recipe_defaults_and_unknown():
    assert get_recipe(None).name == "supervised"
    assert get_recipe("supervised").name == "supervised"
    assert get_recipe("") .name == "supervised"  # empty string → default
    assert get_recipe("gan") is None  # not registered yet (Phase E)
    assert DEFAULT_RECIPE == "supervised"


# --- registry integrity ----------------------------------------------------

def test_registry_entries_are_well_formed():
    assert "supervised" in RECIPES
    for r in RECIPES.values():
        assert r.name and r.label
        assert r.roles, f"{r.name} has no roles"
        assert all(role.role and role.label for role in r.roles)
        # role_params keys must be declared roles.
        role_names = {role.role for role in r.roles}
        assert set(r.role_params).issubset(role_names)
        assert callable(r.generate)


def test_supervised_shape():
    s = RECIPES["supervised"]
    assert [role.role for role in s.roles] == ["model"]
    assert s.needs_targets is True and s.has_val is True
    assert {p.name for p in s.params} >= {"loss", "optimizer", "lr", "epochs", "device"}


# --- API + runner routing ---------------------------------------------------

def test_recipes_endpoint_payload():
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:
        recipes = c.get("/api/recipes").json()
    sup = next(r for r in recipes if r["name"] == "supervised")
    assert sup["roles"] == [{"role": "model", "label": "Model"}]
    assert sup["needs_targets"] is True and sup["has_val"] is True
    # The generator function is backend-only — never serialized.
    assert "generate" not in sup
    # Device choices are resolved live (at least auto/cpu present).
    device_param = next(p for p in sup["params"] if p["name"] == "device")
    assert "cpu" in device_param["choices"]


def test_runner_rejects_an_unknown_recipe():
    from backend.runner import RunManager

    g = _classifier({"recipe": "does-not-exist", "epochs": 1})
    mgr = RunManager()
    error = mgr.start(g, namespace={}, emit=lambda m: None)
    assert error is not None and "does-not-exist" in error
