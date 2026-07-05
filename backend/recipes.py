"""Training recipes: declarative loop templates, one generator each.

A recipe is data (roles, form params, a data contract) plus a single
``generate(project)`` function that emits the ``train()`` source shown in the
Training tab and executed by the runner — the same "the app runs exactly the
code it shows" contract the model/data codegen already honors. The runner and
the Training-tab form are generic over the registry, exactly like the node
registry (``backend/registry.py``): adding a loop (adversarial, and later RL)
is one ``RecipeDef``, never a branch in an engine.

``supervised`` is the classic loop, and its generator is literally today's
``codegen.generate_training`` — so single-model output stays byte-identical
(pinned by a golden test). ``generate`` takes the whole :class:`Project` so a
multi-model recipe (GAN, Phase E) can read every model's graph; the supervised
recipe just uses the sole model's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .codegen import generate_training
from .registry import TRAINING_PARAMS, ParamDef
from .schema import ModelDef, Project, graph_from_project


@dataclass(frozen=True)
class RoleDef:
    """A model slot a recipe trains, e.g. the supervised ``model`` or a GAN's
    ``generator``/``discriminator``. ``role`` keys the assignment (role →
    model_id) and the generated ``train()`` argument name."""

    role: str
    label: str


@dataclass(frozen=True)
class RecipeDef:
    """One training loop, described as data + a generator. ``params`` are
    loop-level form fields (epochs, device, …); ``role_params`` are per-role
    fields (a GAN's per-model optimizer/lr). ``needs_targets``/``has_val`` are
    the data contract the Data tab and runner honor (supervised needs y and a
    val split; an adversarial loop does not). ``generate`` emits the ``train()``
    source for a whole project."""

    name: str
    label: str
    roles: list[RoleDef]
    params: list[ParamDef]
    role_params: dict[str, list[ParamDef]]
    needs_targets: bool
    has_val: bool
    # The role whose model receives the real data X (checked by the Data tab's
    # diagnostics): the supervised ``model``, or a GAN's ``discriminator``.
    data_role: str
    generate: Callable[[Project], str]
    # Invoke the generated ``train`` with the built models mapped by role — the
    # one place a recipe's call signature lives, so the runner stays generic over
    # ``(train_fn, models, train_loader, val_loader, on_epoch) -> history``.
    bind: Callable[..., Any]


def _supervised_generate(project: Project) -> str:
    """The classic loop. The sole model's graph carries the project-level
    training config back onto it, so the emitted source is byte-identical to
    ``generate_training(graph)`` for a single-model project."""
    return generate_training(graph_from_project(project))


def _supervised_bind(train, models, train_loader, val_loader, on_epoch):
    return train(models["model"], train_loader, val_loader=val_loader, on_epoch=on_epoch)


SUPERVISED = RecipeDef(
    name="supervised",
    label="Supervised",
    roles=[RoleDef("model", "Model")],
    params=list(TRAINING_PARAMS),
    role_params={},
    needs_targets=True,
    has_val=True,
    data_role="model",
    generate=_supervised_generate,
    bind=_supervised_bind,
)


# --- GAN (adversarial) ------------------------------------------------------

# Loop-level knobs (epochs/device/seed/autosave). No loss/metric: a GAN's loss
# is fixed (BCE on the real/fake decision) and there's no held-out accuracy.
GAN_PARAMS: list[ParamDef] = [
    ParamDef("epochs", "Epochs", "int", 20),
    ParamDef("device", "Device", "enum", "auto", choices=["auto", "cpu"]),
    ParamDef("seed", "Seed", "int", None, optional=True),
    ParamDef("autosave_every", "Autosave Every (epochs)", "int", None, optional=True),
]

# Each of the two models gets its own optimizer knob.
GAN_ROLE_PARAMS: list[ParamDef] = [ParamDef("lr", "Learning Rate", "float", 2e-4)]


def _model_by_id(project: Project, model_id: str | None) -> ModelDef | None:
    return next((m for m in project.models if m.id == model_id), None)


def _latent_dims(model: ModelDef | None) -> list[int]:
    """The generator's noise vector shape (its Input node's dims, batch dropped).
    Falls back to [100] when the role/Input isn't resolvable yet (e.g. a preview
    before roles are assigned)."""
    if model is None:
        return [100]
    inp = next((n for n in model.graph.nodes if n.type == "Input"), None)
    if inp is None:
        return [100]
    dims = [int(t) for t in str(inp.params.get("shape", "1, 100")).split(",") if t.strip()]
    return dims[1:] or [100]


def _gan_latent_dims(project: Project) -> list[int]:
    """The generator's latent size — from a noise node wired into it (the
    explicit source of truth), or its Input shape as a fallback before a noise
    node is provisioned."""
    roles = (project.training or {}).get("roles") or {}
    gen_id = roles.get("generator")
    for link in project.links:
        if link.source_data is not None and link.target_model == gen_id:
            dn = next(
                (d for d in project.data_nodes if d.id == link.source_data and d.kind == "noise"), None
            )
            if dn is not None:
                dims = [int(t) for t in str((dn.config or {}).get("dims", "")).split(",") if t.strip()]
                if dims:
                    return dims
    return _latent_dims(_model_by_id(project, gen_id))


def _gan_generate(project: Project) -> str:
    """The adversarial loop: per batch, one discriminator step (real→1, fake→0)
    then one generator step (fool the discriminator), tracking g_loss/d_loss.
    Latent noise is drawn to the wired noise node's shape (or the generator's
    Input as a fallback). Emits
    ``train(generator, discriminator, loader, *, device, on_epoch)``."""
    training = project.training or {}
    epochs = int(training.get("epochs", 20))
    device = str(training.get("device", "auto"))
    per_role = training.get("per_role") or {}
    g_lr = float((per_role.get("generator") or {}).get("lr", 2e-4))
    d_lr = float((per_role.get("discriminator") or {}).get("lr", 2e-4))
    noise = ", ".join(str(d) for d in _gan_latent_dims(project))

    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        f"def train(generator, discriminator, loader, *, device={device!r}, on_epoch=None):",
        '    if device == "auto":',
        "        if torch.cuda.is_available():",
        '            device = "cuda"',
        '        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():',
        '            device = "mps"',
        "        else:",
        '            device = "cpu"',
        "    device = torch.device(device)",
        "    generator = generator.to(device)",
        "    discriminator = discriminator.to(device)",
        "    criterion = nn.BCEWithLogitsLoss()",
        f"    opt_g = torch.optim.Adam(generator.parameters(), lr={g_lr!r}, betas=(0.5, 0.999))",
        f"    opt_d = torch.optim.Adam(discriminator.parameters(), lr={d_lr!r}, betas=(0.5, 0.999))",
        '    history = {"g_loss": [], "d_loss": []}',
        f"    for epoch in range({epochs}):",
        "        g_running, d_running, batches = 0.0, 0.0, 0",
        "        for batch in loader:",
        "            real = batch[0].to(device)",
        "            n = real.size(0)",
        f"            noise = torch.randn(n, {noise}, device=device)",
        "            fake = generator(noise)",
        "            # Discriminator step: score real as 1, fake as 0.",
        "            opt_d.zero_grad()",
        "            d_real = discriminator(real)",
        "            d_fake = discriminator(fake.detach())",
        "            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))",
        "            d_loss.backward()",
        "            opt_d.step()",
        "            # Generator step: push the discriminator toward calling fakes real.",
        "            opt_g.zero_grad()",
        "            g_out = discriminator(fake)",
        "            g_loss = criterion(g_out, torch.ones_like(g_out))",
        "            g_loss.backward()",
        "            opt_g.step()",
        "            d_running += d_loss.item()",
        "            g_running += g_loss.item()",
        "            batches += 1",
        '        history["g_loss"].append(g_running / batches)',
        '        history["d_loss"].append(d_running / batches)',
        f'        print(f"epoch {{epoch + 1}}/{epochs}  g_loss {{history[\'g_loss\'][-1]:.4f}}  d_loss {{history[\'d_loss\'][-1]:.4f}}")',
        "        if on_epoch is not None and on_epoch(epoch + 1, history) is False:",
        "            break",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


def _gan_bind(train, models, train_loader, val_loader, on_epoch):
    # val_loader is unused (a GAN has no held-out split); the recipe declares
    # has_val=False so the data pipeline never builds one.
    return train(models["generator"], models["discriminator"], train_loader, on_epoch=on_epoch)


GAN = RecipeDef(
    name="gan",
    label="GAN (adversarial)",
    roles=[RoleDef("generator", "Generator"), RoleDef("discriminator", "Discriminator")],
    params=GAN_PARAMS,
    role_params={"generator": GAN_ROLE_PARAMS, "discriminator": GAN_ROLE_PARAMS},
    needs_targets=False,
    has_val=False,
    data_role="discriminator",  # real images feed the discriminator
    generate=_gan_generate,
    bind=_gan_bind,
)


RECIPES: dict[str, RecipeDef] = {SUPERVISED.name: SUPERVISED, GAN.name: GAN}

DEFAULT_RECIPE = "supervised"


def get_recipe(name: str | None) -> RecipeDef | None:
    """The recipe by name, defaulting to supervised for an unset name. Returns
    None for an unknown name (the caller surfaces a clear error)."""
    return RECIPES.get(name or DEFAULT_RECIPE)
