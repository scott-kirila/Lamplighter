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
    assert get_recipe("gan").name == "gan"
    assert get_recipe("does-not-exist") is None
    assert DEFAULT_RECIPE == "supervised"


# --- the GAN loop generates and trains -------------------------------------

def _gan_project(epochs=3):
    from backend.schema import Graph, ModelDef, Project

    gen = graph(
        [node("in", "Input", {"shape": "1, 100"}), node("l", "Linear", {"out_features": 8}), node("out", "Output")],
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
            "recipe": "gan",
            "epochs": epochs,
            "roles": {"generator": "g", "discriminator": "d"},
            "per_role": {"generator": {"lr": 0.01}, "discriminator": {"lr": 0.01}},
        },
    )


def test_gan_generate_runs_and_moves_both_models():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    src = RECIPES["gan"].generate(_gan_project(epochs=3))
    assert "torch.randn(n, 100, device=device)" in src  # latent from the generator's Input
    assert "BCEWithLogitsLoss" in src

    ns: dict = {}
    exec(compile(src, "<gan>", "exec"), ns)  # noqa: S102
    train = ns["train"]

    generator = nn.Sequential(nn.Linear(100, 8))
    discriminator = nn.Sequential(nn.Linear(8, 1))
    before_g = [p.detach().clone() for p in generator.parameters()]
    before_d = [p.detach().clone() for p in discriminator.parameters()]
    loader = DataLoader(TensorDataset(torch.randn(24, 8)), batch_size=8)

    seen: list[int] = []
    history = train(generator, discriminator, loader, device="cpu", on_epoch=lambda e, h: seen.append(e))

    assert len(history["g_loss"]) == 3 and len(history["d_loss"]) == 3
    assert all(v == v for v in history["g_loss"] + history["d_loss"])  # finite, no NaN
    assert any(not torch.equal(b, p) for b, p in zip(before_g, generator.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(before_d, discriminator.parameters()))
    assert seen == [1, 2, 3]  # on_epoch fired each epoch


def test_gan_latent_comes_from_a_wired_noise_node():
    from backend.schema import DataNode, ModelLink

    project = _gan_project(epochs=2)
    # A noise node (dims 50) wired into the generator is the latent source of
    # truth — it overrides the generator's Input shape.
    project.data_nodes = [DataNode(id="z", kind="noise", name="Noise", config={"dims": "50"})]
    project.links = [ModelLink(id="L", source_data="z", target_model="g")]
    assert "torch.randn(n, 50, device=device)" in RECIPES["gan"].generate(project)


def test_gan_latent_falls_back_to_the_generator_input():
    # No noise node → the generator's Input (100) drives the latent (compat).
    assert "torch.randn(n, 100, device=device)" in RECIPES["gan"].generate(_gan_project(epochs=2))


def test_gan_on_epoch_can_stop_early():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    src = RECIPES["gan"].generate(_gan_project(epochs=10))
    ns: dict = {}
    exec(compile(src, "<gan>", "exec"), ns)  # noqa: S102
    loader = DataLoader(TensorDataset(torch.randn(16, 8)), batch_size=8)
    history = ns["train"](
        nn.Sequential(nn.Linear(100, 8)), nn.Sequential(nn.Linear(8, 1)), loader,
        on_epoch=lambda e, h: e < 2,  # stop after epoch 2
    )
    assert len(history["g_loss"]) == 2


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
