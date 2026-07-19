"""The training recipe registry.

Pins the recipe abstraction that makes the GAN loop (Phase E) a data addition:
the supervised recipe must generate byte-identical source to today's
``generate_training`` (so the classic single-model flow is untouched), the
registry must be well-formed, and the runner/API must route through it.
"""
from lamplighter.backend.codegen import generate_training
from lamplighter.backend.recipes import DEFAULT_RECIPE, RECIPES, get_recipe
from tests.helpers import edge, graph, node, single_model_project


def _classifier(training=None, data=None):
    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("a", "Linear", {"out_features": 16}),
         node("r", "ReLU"), node("b", "Linear", {"out_features": 3}), node("out", "Output")],
        [edge("in", "a"), edge("a", "r"), edge("r", "b"), edge("b", "out")],
    )
    return single_model_project(g, training=training, data=data)


def _multi_input():
    g = graph(
        [node("x0", "Input", {"shape": "1, 4"}), node("x1", "Input", {"shape": "1, 6"}),
         node("cat", "Concat", {"dim": 1}), node("l", "Linear", {"out_features": 2}), node("out", "Output")],
        [edge("x0", "cat", tgt_h="in0"), edge("x1", "cat", tgt_h="in1"), edge("cat", "l"), edge("l", "out")],
    )
    return single_model_project(g, training={"loss": "MSELoss", "metric": "none", "epochs": 7}, data={"val_split": 0.2})


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
    for proj in cases:
        assert gen(proj) == generate_training(proj.models[0].graph, proj.training)


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
    from lamplighter.backend.schema import Graph, ModelDef, Project

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
    from lamplighter.backend.schema import DataNode, ModelLink

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


# --- the conditional-GAN loop generates with wiring-driven arg order --------

def _cgan_project(epochs=3, label_first=False):
    from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project

    # Canvas y-order decides forward()-arg order; the y-pin wire decides which
    # port is the label regardless of that order.
    noise_y, label_y = (1.0, 0.0) if label_first else (0.0, 1.0)
    gen = graph(
        [node("noise", "Input", {"shape": "1, 100"}, y=noise_y),
         node("glabel", "Input", {"shape": "1", "dtype": "long"}, y=label_y),
         node("gcat", "Concat", {"dim": 1}), node("gl", "Linear", {"out_features": 8}), node("gout", "Output")],
        [edge("noise", "gcat", tgt_h="in0"), edge("glabel", "gcat", tgt_h="in1"),
         edge("gcat", "gl"), edge("gl", "gout")],
    )
    disc = graph(
        [node("image", "Input", {"shape": "1, 8"}, y=0.0),
         node("dlabel", "Input", {"shape": "1", "dtype": "long"}, y=1.0),
         node("dcat", "Concat", {"dim": 1}), node("dl", "Linear", {"out_features": 1}), node("dout", "Output")],
        [edge("image", "dcat", tgt_h="in0"), edge("dlabel", "dcat", tgt_h="in1"),
         edge("dcat", "dl"), edge("dl", "dout")],
    )
    return Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        data_nodes=[
            DataNode(id="z", kind="noise", name="Noise", config={"dims": "100"}),
            DataNode(id="ds", kind="dataset", name="MNIST",
                     config={"source": "memory", "x_var": "X", "y_var": "Y"}),
        ],
        # noise → gen.noise, label(y) → gen.label, image(x) → disc.image, label(y) → disc.label.
        links=[
            ModelLink(id="l1", source_data="z", target_model="g", target_input="noise"),
            ModelLink(id="l2", source_data="ds", source_pin="y", target_model="g", target_input="glabel"),
            ModelLink(id="l3", source_data="ds", source_pin="x", target_model="d", target_input="image"),
            ModelLink(id="l4", source_data="ds", source_pin="y", target_model="d", target_input="dlabel"),
        ],
        training={
            "recipe": "cgan", "epochs": epochs,
            "roles": {"generator": "g", "discriminator": "d"},
            "per_role": {"generator": {"lr": 0.01}, "discriminator": {"lr": 0.01}},
        },
    )


def test_cgan_generate_conditions_both_models_in_wired_arg_order():
    src = RECIPES["cgan"].generate(_cgan_project(epochs=3))
    assert "for images, labels in loader:" in src
    assert "torch.randn(n, 100, device=device)" in src  # latent from the wired noise node
    # noise before label on the canvas → generator(noise, labels).
    assert "fake = generator(noise, labels)" in src
    # image before label → discriminator(<img>, labels) at each of the three calls.
    assert "d_real = discriminator(real, labels)" in src
    assert "d_fake = discriminator(fake.detach(), labels)" in src
    assert "g_out = discriminator(fake, labels)" in src


def test_cgan_arg_order_follows_the_wiring_not_placement():
    # Put the label Input first on the canvas: the y-pin wire still marks it the
    # label, so the emitted call is generator(labels, noise) — order from the graph.
    src = RECIPES["cgan"].generate(_cgan_project(epochs=2, label_first=True))
    assert "fake = generator(labels, noise)" in src


def test_cgan_generate_runs_and_moves_both_models():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    src = RECIPES["cgan"].generate(_cgan_project(epochs=3))
    ns: dict = {}
    exec(compile(src, "<cgan>", "exec"), ns)  # noqa: S102
    train = ns["train"]

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.fc = nn.Embedding(10, 4), nn.Linear(104, 8)

        def forward(self, noise, labels):
            return self.fc(torch.cat([noise, self.emb(labels)], dim=1))

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.fc = nn.Embedding(10, 4), nn.Linear(12, 1)

        def forward(self, image, labels):
            return self.fc(torch.cat([image, self.emb(labels)], dim=1))

    generator, discriminator = G(), D()
    before_g = [p.detach().clone() for p in generator.parameters()]
    before_d = [p.detach().clone() for p in discriminator.parameters()]
    loader = DataLoader(TensorDataset(torch.randn(24, 8), torch.randint(0, 10, (24,))), batch_size=8)

    seen: list[int] = []
    history = train(generator, discriminator, loader, device="cpu", on_epoch=lambda e, h: seen.append(e))

    assert len(history["g_loss"]) == 3 and len(history["d_loss"]) == 3
    assert all(v == v for v in history["g_loss"] + history["d_loss"])  # finite
    assert any(not torch.equal(b, p) for b, p in zip(before_g, generator.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(before_d, discriminator.parameters()))
    assert seen == [1, 2, 3]


def test_cgan_shape():
    c = RECIPES["cgan"]
    assert [role.role for role in c.roles] == ["generator", "discriminator"]
    assert c.needs_targets is True and c.has_val is False and c.data_role == "discriminator"


# --- loops put their models in train mode (BN/Dropout hygiene) --------------

def test_gan_and_cgan_loops_switch_models_to_train_mode():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # GAN: hand the loop eval-mode models; it must flip both to train so a
    # generator carrying BatchNorm/Dropout trains correctly. Read .training
    # inside on_epoch — a point that only executes once the loop is running.
    ns: dict = {}
    exec(compile(RECIPES["gan"].generate(_gan_project(epochs=1)), "<gan>", "exec"), ns)  # noqa: S102
    g, d = nn.Sequential(nn.Linear(100, 8)).eval(), nn.Sequential(nn.Linear(8, 1)).eval()
    loader = DataLoader(TensorDataset(torch.randn(16, 8)), batch_size=8)
    modes: list = []
    ns["train"](g, d, loader, device="cpu", on_epoch=lambda e, h: modes.append((g.training, d.training)))
    assert modes == [(True, True)]

    # cGAN: same guarantee for the conditional models.
    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.fc = nn.Embedding(10, 4), nn.Linear(104, 8)

        def forward(self, noise, labels):
            return self.fc(torch.cat([noise, self.emb(labels)], dim=1))

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.fc = nn.Embedding(10, 4), nn.Linear(12, 1)

        def forward(self, image, labels):
            return self.fc(torch.cat([image, self.emb(labels)], dim=1))

    ns = {}
    exec(compile(RECIPES["cgan"].generate(_cgan_project(epochs=1)), "<cgan>", "exec"), ns)  # noqa: S102
    g, d = G().eval(), D().eval()
    loader = DataLoader(TensorDataset(torch.randn(16, 8), torch.randint(0, 10, (16,))), batch_size=8)
    modes = []
    ns["train"](g, d, loader, device="cpu", on_epoch=lambda e, h: modes.append((g.training, d.training)))
    assert modes == [(True, True)]


def test_vae_loop_switches_models_to_train_mode():
    from collections import namedtuple

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from tests.test_vae_run import _vae_project

    Enc_out = namedtuple("Enc_out", ["mu", "logvar"])

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.mu_h, self.lv_h = nn.Linear(8, 3), nn.Linear(8, 3)

        def forward(self, x):
            return Enc_out(self.mu_h(x), self.lv_h(x))

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 8)

        def forward(self, z):
            return torch.sigmoid(self.fc(z))

    ns: dict = {}
    exec(compile(RECIPES["vae"].generate(_vae_project(epochs=1)), "<vae>", "exec"), ns)  # noqa: S102
    enc, dec = Enc().eval(), Dec().eval()
    loader = DataLoader(TensorDataset(torch.rand(16, 8)), batch_size=8)
    modes: list = []
    ns["train"](enc, dec, loader, device="cpu", on_epoch=lambda e, h: modes.append((enc.training, dec.training)))
    assert modes == [(True, True)]


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

    from lamplighter.backend.app import app

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
    from lamplighter.backend.runner import RunManager

    proj = _classifier({"recipe": "does-not-exist", "epochs": 1})
    mgr = RunManager()
    error = mgr.start(proj, namespace={}, emit=lambda m: None)
    assert error is not None and "does-not-exist" in error
