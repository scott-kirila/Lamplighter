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

from .codegen import device_resolve_lines, generate_training, model_inputs
from .inference import build_incoming
from .registry import TRAINING_PARAMS, ParamDef
from .schema import Graph, ModelDef, Project


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
    # ``(train_fn, models, train_loader, val_loader, on_epoch, on_step) -> history``.
    # ``on_step`` (per-batch loss) is currently threaded only by the supervised
    # loop; the adversarial/VAE loops accept and ignore it (no per-step yet).
    bind: Callable[..., Any]
    # The recipe's data source: "loader" (the default — tensors/datasets through
    # the generated make_dataloaders) or "env" (an RL recipe — a Gymnasium
    # environment node wired into the data_role model; the env is created
    # INSIDE the generated train(), so no loader path runs at all).
    data: str = "loader"


def _supervised_generate(project: Project) -> str:
    """The classic loop: the sole model's graph for shape, the project's training
    config for the loop — emitted straight, no graph round-trip."""
    graph = project.models[0].graph if project.models else Graph()
    return generate_training(graph, project.training)


def _supervised_bind(train, models, train_loader, val_loader, on_epoch, on_step=None):
    return train(models["model"], train_loader, val_loader=val_loader, on_epoch=on_epoch, on_step=on_step)


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
        f"def train(generator, discriminator, loader, *, device={device!r}, on_epoch=None, on_step=None):",
        *device_resolve_lines(),
        "    generator = generator.to(device)",
        "    discriminator = discriminator.to(device)",
        "    generator.train()",
        "    discriminator.train()",
        "    criterion = nn.BCEWithLogitsLoss()",
        f"    opt_g = torch.optim.Adam(generator.parameters(), lr={g_lr!r}, betas=(0.5, 0.999))",
        f"    opt_d = torch.optim.Adam(discriminator.parameters(), lr={d_lr!r}, betas=(0.5, 0.999))",
        '    history = {"g_loss": [], "d_loss": []}',
        "    step = 0",
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
        "            dl, gl = d_loss.item(), g_loss.item()",
        "            d_running += dl",
        "            g_running += gl",
        "            batches += 1",
        "            step += 1",
        "            if on_step is not None:",
        "                on_step(step, {'g_loss': gl, 'd_loss': dl})",
        '        history["g_loss"].append(g_running / batches)',
        '        history["d_loss"].append(d_running / batches)',
        "        if on_epoch is None:",
        f'            print(f"epoch {{epoch + 1}}/{epochs}  g_loss {{history[\'g_loss\'][-1]:.4f}}  d_loss {{history[\'d_loss\'][-1]:.4f}}")',
        "        if on_epoch is not None and on_epoch(epoch + 1, history) is False:",
        "            break",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


def _gan_bind(train, models, train_loader, val_loader, on_epoch, on_step=None):
    # val_loader is unused (a GAN has no held-out split); the recipe declares
    # has_val=False so the data pipeline never builds one.
    return train(models["generator"], models["discriminator"], train_loader, on_epoch=on_epoch, on_step=on_step)


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


# --- Conditional GAN --------------------------------------------------------

# A cGAN reuses the GAN's loop-level and per-role knobs; the only difference is
# that a class label conditions both models (fed from the dataset's y pin).
CGAN_PARAMS: list[ParamDef] = list(GAN_PARAMS)
CGAN_ROLE_PARAMS: list[ParamDef] = list(GAN_ROLE_PARAMS)


def _cond_ports(project: Project, model_id: str | None) -> tuple[list[str], str | None]:
    """A conditional model's Input node ids in forward()-arg order, plus which one
    is the label port — the Input wired from the dataset's ``y`` pin (Option-A
    explicit wiring). Falls back to the last input when nothing is label-wired
    (e.g. a preview before auto-provisioning), so generation always resolves."""
    model = _model_by_id(project, model_id)
    if model is None:
        return [], None
    graph = model.graph
    node_map = {n.id: n for n in graph.nodes}
    ordered = model_inputs(graph, build_incoming(graph), node_map)
    label_id = next(
        (
            link.target_input
            for link in project.links
            if link.source_data is not None
            and link.target_model == model_id
            and link.source_pin == "y"
            and link.target_input in ordered
        ),
        None,
    )
    if label_id is None and len(ordered) > 1:
        label_id = ordered[-1]
    return ordered, label_id


def _cond_args(ordered: list[str], label_id: str | None, primary: str) -> str:
    """The positional args for a conditional model's forward call: ``labels`` at
    the label port, ``primary`` (the noise or the image expr) everywhere else."""
    return ", ".join("labels" if nid == label_id else primary for nid in ordered)


def _cgan_generate(project: Project) -> str:
    """The conditional adversarial loop: same D-then-G structure as the GAN, but a
    class label (the dataset's ``y``) conditions both models. The label reaches
    each model at the port wired from the dataset's y pin — generation reads that
    wiring so the emitted ``generator(noise, labels)`` / ``discriminator(real,
    labels)`` calls match each model's forward-arg order regardless of placement.
    Fake batches condition on the same real labels (standard cGAN)."""
    training = project.training or {}
    epochs = int(training.get("epochs", 20))
    device = str(training.get("device", "auto"))
    per_role = training.get("per_role") or {}
    g_lr = float((per_role.get("generator") or {}).get("lr", 2e-4))
    d_lr = float((per_role.get("discriminator") or {}).get("lr", 2e-4))
    noise = ", ".join(str(d) for d in _gan_latent_dims(project))

    roles = training.get("roles") or {}
    g_ordered, g_label = _cond_ports(project, roles.get("generator"))
    d_ordered, d_label = _cond_ports(project, roles.get("discriminator"))
    gen_call = _cond_args(g_ordered, g_label, "noise") or "noise"
    d_real_call = _cond_args(d_ordered, d_label, "real") or "real"
    d_fake_call = _cond_args(d_ordered, d_label, "fake.detach()") or "fake.detach()"
    g_fake_call = _cond_args(d_ordered, d_label, "fake") or "fake"

    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        f"def train(generator, discriminator, loader, *, device={device!r}, on_epoch=None, on_step=None):",
        *device_resolve_lines(),
        "    generator = generator.to(device)",
        "    discriminator = discriminator.to(device)",
        "    generator.train()",
        "    discriminator.train()",
        "    criterion = nn.BCEWithLogitsLoss()",
        f"    opt_g = torch.optim.Adam(generator.parameters(), lr={g_lr!r}, betas=(0.5, 0.999))",
        f"    opt_d = torch.optim.Adam(discriminator.parameters(), lr={d_lr!r}, betas=(0.5, 0.999))",
        '    history = {"g_loss": [], "d_loss": []}',
        "    step = 0",
        f"    for epoch in range({epochs}):",
        "        g_running, d_running, batches = 0.0, 0.0, 0",
        "        for images, labels in loader:",
        "            real = images.to(device)",
        "            labels = labels.to(device)",
        "            n = real.size(0)",
        f"            noise = torch.randn(n, {noise}, device=device)",
        f"            fake = generator({gen_call})",
        "            # Discriminator step: score real as 1, fake as 0 (same labels).",
        "            opt_d.zero_grad()",
        f"            d_real = discriminator({d_real_call})",
        f"            d_fake = discriminator({d_fake_call})",
        "            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))",
        "            d_loss.backward()",
        "            opt_d.step()",
        "            # Generator step: push the discriminator toward calling fakes real.",
        "            opt_g.zero_grad()",
        f"            g_out = discriminator({g_fake_call})",
        "            g_loss = criterion(g_out, torch.ones_like(g_out))",
        "            g_loss.backward()",
        "            opt_g.step()",
        "            dl, gl = d_loss.item(), g_loss.item()",
        "            d_running += dl",
        "            g_running += gl",
        "            batches += 1",
        "            step += 1",
        "            if on_step is not None:",
        "                on_step(step, {'g_loss': gl, 'd_loss': dl})",
        '        history["g_loss"].append(g_running / batches)',
        '        history["d_loss"].append(d_running / batches)',
        "        if on_epoch is None:",
        f'            print(f"epoch {{epoch + 1}}/{epochs}  g_loss {{history[\'g_loss\'][-1]:.4f}}  d_loss {{history[\'d_loss\'][-1]:.4f}}")',
        "        if on_epoch is not None and on_epoch(epoch + 1, history) is False:",
        "            break",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


def _cgan_bind(train, models, train_loader, val_loader, on_epoch, on_step=None):
    # val_loader is unused (has_val=False); labels ride the train_loader as its y.
    return train(models["generator"], models["discriminator"], train_loader, on_epoch=on_epoch, on_step=on_step)


CGAN = RecipeDef(
    name="cgan",
    label="Conditional GAN",
    roles=[RoleDef("generator", "Generator"), RoleDef("discriminator", "Discriminator")],
    params=CGAN_PARAMS,
    role_params={"generator": CGAN_ROLE_PARAMS, "discriminator": CGAN_ROLE_PARAMS},
    needs_targets=True,  # the class label rides the loader as its y
    has_val=False,
    data_role="discriminator",  # real images (X) feed the discriminator
    generate=_cgan_generate,
    bind=_cgan_bind,
)


# --- VAE (variational autoencoder) ------------------------------------------

# Joint training: one optimizer spans both models (unlike a GAN's two), so lr
# is a loop-level knob and there are no per-role params. beta scales the KL
# term (beta-VAE); recon picks the reconstruction loss.
VAE_PARAMS: list[ParamDef] = [
    ParamDef("epochs", "Epochs", "int", 20),
    ParamDef("lr", "Learning Rate", "float", 1e-3),
    ParamDef("beta", "Beta (KL weight)", "float", 1.0),
    ParamDef("recon", "Reconstruction Loss", "enum", "bce", choices=["bce", "mse"]),
    ParamDef("device", "Device", "enum", "auto", choices=["auto", "cpu"]),
    ParamDef("seed", "Seed", "int", None, optional=True),
    ParamDef("autosave_every", "Autosave Every (epochs)", "int", None, optional=True),
]


def _vae_check_encoder(project: Project) -> None:
    """The recipe reads the encoder's outputs BY NAME (mu/logvar), so canvas
    Output order can't matter — enforce the naming with a clear error. Skipped
    when the role isn't assigned yet (a preview before role assignment)."""
    roles = (project.training or {}).get("roles") or {}
    encoder = _model_by_id(project, roles.get("encoder"))
    if encoder is None:
        return
    names = {
        str(n.params.get("name", "") or "").strip()
        for n in encoder.graph.nodes
        if n.type == "Output"
    }
    if not {"mu", "logvar"} <= names:
        raise ValueError(
            "the VAE encoder needs two Output nodes named 'mu' and 'logvar' "
            f"(found: {', '.join(sorted(n for n in names if n)) or 'unnamed outputs'})"
        )


def _vae_generate(project: Project) -> str:
    """The VAE loop: encode → reparameterize (z = mu + eps·exp(logvar/2)) →
    decode → reconstruction + beta·KL, one optimizer over both models. The
    per-sample loss sums over features and averages over the batch (the
    standard normalization, so beta means the same thing at any batch size)."""
    _vae_check_encoder(project)
    training = project.training or {}
    epochs = int(training.get("epochs", 20))
    device = str(training.get("device", "auto"))
    lr = float(training.get("lr", 1e-3))
    beta = float(training.get("beta", 1.0))
    recon = str(training.get("recon", "bce"))
    if recon not in ("bce", "mse"):
        raise ValueError(f"unknown reconstruction loss '{recon}' — expected bce or mse")
    recon_call = (
        'F.binary_cross_entropy(x_hat, real, reduction="sum")'
        if recon == "bce"
        else 'F.mse_loss(x_hat, real, reduction="sum")'
    )

    lines = [
        "import torch",
        "import torch.nn.functional as F",
        "",
        "",
        f"def train(encoder, decoder, loader, *, device={device!r}, on_epoch=None, on_step=None):",
        *device_resolve_lines(),
        "    encoder = encoder.to(device)",
        "    decoder = decoder.to(device)",
        "    encoder.train()",
        "    decoder.train()",
        f"    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr={lr!r})",
        '    history = {"recon_loss": [], "kl_loss": []}',
        "    step = 0",
        f"    for epoch in range({epochs}):",
        "        recon_running, kl_running, seen = 0.0, 0.0, 0",
        "        for batch in loader:",
        "            real = batch[0].to(device)",
        "            n = real.size(0)",
        "            enc = encoder(real)",
        "            mu, logvar = enc.mu, enc.logvar",
        "            # Reparameterization: sample z differentiably.",
        "            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)",
        "            x_hat = decoder(z)",
        f"            recon_loss = {recon_call} / n",
        "            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n",
        f"            loss = recon_loss + {beta!r} * kl_loss",
        "            opt.zero_grad()",
        "            loss.backward()",
        "            opt.step()",
        "            rl, kl = recon_loss.item(), kl_loss.item()",
        "            recon_running += rl * n",
        "            kl_running += kl * n",
        "            seen += n",
        "            step += 1",
        "            if on_step is not None:",
        "                on_step(step, {'recon_loss': rl, 'kl_loss': kl})",
        '        history["recon_loss"].append(recon_running / seen)',
        '        history["kl_loss"].append(kl_running / seen)',
        "        if on_epoch is None:",
        f'            print(f"epoch {{epoch + 1}}/{epochs}  recon {{history[\'recon_loss\'][-1]:.4f}}  kl {{history[\'kl_loss\'][-1]:.4f}}")',
        "        if on_epoch is not None and on_epoch(epoch + 1, history) is False:",
        "            break",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


def _vae_bind(train, models, train_loader, val_loader, on_epoch, on_step=None):
    # val_loader is unused (has_val=False — reconstruction has no held-out split v1).
    return train(models["encoder"], models["decoder"], train_loader, on_epoch=on_epoch, on_step=on_step)


VAE = RecipeDef(
    name="vae",
    label="VAE (autoencoder)",
    roles=[RoleDef("encoder", "Encoder"), RoleDef("decoder", "Decoder")],
    params=VAE_PARAMS,
    role_params={},
    needs_targets=False,  # reconstruction: the input is the target
    has_val=False,
    data_role="encoder",  # real samples feed the encoder
    generate=_vae_generate,
    bind=_vae_bind,
)


# --- REINFORCE (policy gradient, RL) ----------------------------------------

# The curated discrete classic-control environments (dependency-free renders,
# small obs spaces). The id lands in the source as gym.make's argument —
# validated against this list (the torchvision-dataset-name rule) so a raw API
# caller can't smuggle an arbitrary string AND the failure happens pre-flight.
# Curation rule: every listed env must be learnable from random init by the
# offered recipes. MountainCar-v0 fails it — ~0% random success within its
# 200-step limit means every return is exactly -200, so the group-relative
# advantage is identically zero and the gradient is pure noise (it's the
# canonical hard-EXPLORATION benchmark; it can return with an off-policy/DQN
# recipe, whose replay can reuse a rare success).
RL_ENVS = ("CartPole-v1", "Acrobot-v1")

# "epochs" keeps its cfg name (the runner/resume machinery is keyed on it) but
# reads as iterations — one iteration = a batch of episodes + one update.
REINFORCE_PARAMS: list[ParamDef] = [
    ParamDef("epochs", "Iterations", "int", 40),
    ParamDef("episodes_per_iter", "Episodes / Iteration", "int", 8),
    ParamDef("lr", "Learning Rate", "float", 1e-2),
    ParamDef("gamma", "Discount (gamma)", "float", 0.99),
    ParamDef("entropy_beta", "Entropy Bonus", "float", 0.01),
    # CPU by default and on purpose: envs step on the CPU and the policy is
    # tiny, so shuttling per-step tensors to an accelerator is a net loss.
    ParamDef("device", "Device", "enum", "cpu", choices=["cpu", "auto"]),
    ParamDef("seed", "Seed", "int", None, optional=True),
    ParamDef("autosave_every", "Autosave Every (iters)", "int", None, optional=True),
]


def _rl_env_id(project: Project) -> str:
    """The wired environment's id — the policy model's env node is the single
    source of truth (the dataset-node rule). Clear errors when unwired or
    outside the curated list. Shared by every policy-role RL recipe."""
    from .schema import resolve_env_config

    roles = (project.training or {}).get("roles") or {}
    policy_id = roles.get("policy") or (project.models[0].id if project.models else None)
    config = resolve_env_config(project, policy_id)
    if config is None:
        raise ValueError(
            "wire an environment into the policy — add one with ＋ env on the Models canvas"
        )
    env_id = str(config.get("env_id", "") or "")
    if env_id not in RL_ENVS:
        raise ValueError(
            f"unknown environment '{env_id}' — expected one of: {', '.join(RL_ENVS)}"
        )
    return env_id


def _reinforce_generate(project: Project) -> str:
    """The policy-gradient loop: per iteration, roll out a batch of episodes
    with the current policy (actions sampled from Categorical over the
    policy's LOGITS — logits-first, no softmax head), compute discounted
    returns-to-go per step, normalize them across the batch (the
    group-relative baseline), and ascend ``logp · advantage`` with an entropy
    bonus. Episodes are seeded from the run's recorded seed (the runner
    injects the resolved value before generation), so rollouts replay."""
    training = project.training or {}
    env_id = _rl_env_id(project)
    epochs = int(training.get("epochs", 40))
    episodes = int(training.get("episodes_per_iter", 8))
    lr = float(training.get("lr", 1e-2))
    gamma = float(training.get("gamma", 0.99))
    beta = float(training.get("entropy_beta", 0.01))
    device = str(training.get("device", "cpu"))
    seed = training.get("seed")
    seed_literal = repr(int(seed)) if seed is not None else "None"

    lines = [
        "import gymnasium as gym",
        "import torch",
        "",
        "",
        f"def train(policy, *, device={device!r}, on_epoch=None, on_step=None):",
        *device_resolve_lines(),
        "    policy = policy.to(device)",
        "    policy.train()",
        f"    env = gym.make({env_id!r})",
        f"    opt = torch.optim.Adam(policy.parameters(), lr={lr!r})",
        f"    base_seed = {seed_literal}  # the run's recorded seed — rollouts replay",
        '    history = {"mean_return": [], "episode_len": [], "policy_loss": [], "entropy": []}',
        "    episode = 0",
        f"    for iteration in range({epochs}):",
        "        log_probs, entropies, step_returns = [], [], []",
        "        ep_returns, ep_lens = [], []",
        f"        for _ in range({episodes}):",
        "            obs, _ = env.reset(seed=None if base_seed is None else base_seed + episode)",
        "            done, rewards = False, []",
        "            while not done:",
        "                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)",
        "                dist = torch.distributions.Categorical(logits=policy(x))",
        "                action = dist.sample()",
        "                log_probs.append(dist.log_prob(action).squeeze(0))",
        "                entropies.append(dist.entropy().squeeze(0))",
        "                obs, reward, terminated, truncated, _ = env.step(int(action))",
        "                rewards.append(float(reward))",
        "                done = terminated or truncated",
        "            # Returns-to-go: each step's credit is its discounted future reward.",
        "            g, rtg = 0.0, []",
        "            for r in reversed(rewards):",
        f"                g = r + {gamma!r} * g",
        "                rtg.append(g)",
        "            step_returns.extend(reversed(rtg))",
        "            ep_returns.append(sum(rewards))",
        "            ep_lens.append(len(rewards))",
        "            episode += 1",
        "            if on_step is not None:",
        '                on_step(episode, {"episode_return": sum(rewards)})',
        "        returns = torch.as_tensor(step_returns, dtype=torch.float32, device=device)",
        "        # Batch-normalized advantages — the group-relative baseline.",
        "        adv = (returns - returns.mean()) / (returns.std() + 1e-8)",
        "        entropy = torch.stack(entropies).mean()",
        f"        loss = -(torch.stack(log_probs) * adv).mean() - {beta!r} * entropy",
        "        opt.zero_grad()",
        "        loss.backward()",
        "        opt.step()",
        '        history["mean_return"].append(sum(ep_returns) / len(ep_returns))',
        '        history["episode_len"].append(sum(ep_lens) / len(ep_lens))',
        '        history["policy_loss"].append(float(loss.item()))',
        '        history["entropy"].append(float(entropy.item()))',
        "        if on_epoch is None:",
        f'            print(f"iter {{iteration + 1}}/{epochs}  return {{history[\'mean_return\'][-1]:.1f}}  entropy {{history[\'entropy\'][-1]:.3f}}")',
        "        if on_epoch is not None and on_epoch(iteration + 1, history) is False:",
        "            break",
        "    env.close()",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


def _reinforce_bind(train, models, train_loader, val_loader, on_epoch, on_step=None):
    # No loaders — the environment lives inside the generated train().
    return train(models["policy"], on_epoch=on_epoch, on_step=on_step)


REINFORCE = RecipeDef(
    name="reinforce",
    label="REINFORCE (policy gradient)",
    roles=[RoleDef("policy", "Policy")],
    params=REINFORCE_PARAMS,
    role_params={},
    needs_targets=False,
    has_val=False,
    data_role="policy",  # the environment feeds the policy
    generate=_reinforce_generate,
    bind=_reinforce_bind,
    data="env",
)


# --- GRPO (Group Relative Policy Optimization, RL) --------------------------

# GRPO over REINFORCE: the same critic-free group-relative baseline (normalize
# returns across the episode group — no value network), PLUS a clipped-surrogate
# objective reused over several update epochs per rollout (the sample-efficiency
# PPO brought, without PPO's separate critic). The KL-to-reference term of the
# original LLM formulation is omitted — control policies train from scratch, so
# there's no reference model to anchor to; the clip alone is the standard
# from-scratch simplification.
GRPO_PARAMS: list[ParamDef] = [
    ParamDef("epochs", "Iterations", "int", 60),
    ParamDef("episodes_per_iter", "Episodes / Iteration", "int", 8),
    ParamDef("lr", "Learning Rate", "float", 3e-3),
    ParamDef("gamma", "Discount (gamma)", "float", 0.99),
    ParamDef("clip", "Clip Epsilon", "float", 0.2),
    ParamDef("update_epochs", "Update Epochs / Iter", "int", 4),
    ParamDef("entropy_beta", "Entropy Bonus", "float", 0.01),
    ParamDef("device", "Device", "enum", "cpu", choices=["cpu", "auto"]),
    ParamDef("seed", "Seed", "int", None, optional=True),
    ParamDef("autosave_every", "Autosave Every (iters)", "int", None, optional=True),
]


def _grpo_generate(project: Project) -> str:
    """The GRPO loop: collect a GROUP of episodes with the current (frozen)
    policy, credit each step with discounted returns-to-go, and take the
    group-relative advantage (normalize across the whole group — the
    critic-free baseline). Then reuse that rollout for several epochs of a
    CLIPPED-surrogate update: the probability ratio to the frozen policy,
    clipped to ``[1-clip, 1+clip]`` so a single batch can't shove the policy
    too far. Actions are sampled from Categorical over the policy's LOGITS
    (logits-first); episodes are seeded from the run's recorded seed."""
    training = project.training or {}
    env_id = _rl_env_id(project)
    epochs = int(training.get("epochs", 60))
    episodes = int(training.get("episodes_per_iter", 8))
    lr = float(training.get("lr", 3e-3))
    gamma = float(training.get("gamma", 0.99))
    clip = float(training.get("clip", 0.2))
    update_epochs = int(training.get("update_epochs", 4))
    beta = float(training.get("entropy_beta", 0.01))
    device = str(training.get("device", "cpu"))
    seed = training.get("seed")
    seed_literal = repr(int(seed)) if seed is not None else "None"

    lines = [
        "import gymnasium as gym",
        "import torch",
        "",
        "",
        f"def train(policy, *, device={device!r}, on_epoch=None, on_step=None):",
        *device_resolve_lines(),
        "    policy = policy.to(device)",
        "    policy.train()",
        f"    env = gym.make({env_id!r})",
        f"    opt = torch.optim.Adam(policy.parameters(), lr={lr!r})",
        f"    base_seed = {seed_literal}  # the run's recorded seed — rollouts replay",
        '    history = {"mean_return": [], "episode_len": [], "policy_loss": [], "entropy": []}',
        "    episode = 0",
        f"    for iteration in range({epochs}):",
        "        # Collect a group of episodes with the current (frozen) policy.",
        "        obs_steps, act_steps, ret_steps = [], [], []",
        "        ep_returns, ep_lens = [], []",
        f"        for _ in range({episodes}):",
        "            obs, _ = env.reset(seed=None if base_seed is None else base_seed + episode)",
        "            done, rewards = False, []",
        "            while not done:",
        "                x = torch.as_tensor(obs, dtype=torch.float32, device=device)",
        "                with torch.no_grad():",
        "                    action = torch.distributions.Categorical(logits=policy(x.unsqueeze(0))).sample()",
        "                obs_steps.append(x)",
        "                act_steps.append(action.squeeze(0))",
        "                obs, reward, terminated, truncated, _ = env.step(int(action))",
        "                rewards.append(float(reward))",
        "                done = terminated or truncated",
        "            g, rtg = 0.0, []",
        "            for r in reversed(rewards):",
        f"                g = r + {gamma!r} * g",
        "                rtg.append(g)",
        "            ret_steps.extend(reversed(rtg))",
        "            ep_returns.append(sum(rewards))",
        "            ep_lens.append(len(rewards))",
        "            episode += 1",
        "            if on_step is not None:",
        '                on_step(episode, {"episode_return": sum(rewards)})',
        "        obs_batch = torch.stack(obs_steps)",
        "        act_batch = torch.stack(act_steps)",
        "        returns = torch.as_tensor(ret_steps, dtype=torch.float32, device=device)",
        "        # Group-relative advantages — the critic-free baseline (no value net).",
        "        adv = (returns - returns.mean()) / (returns.std() + 1e-8)",
        "        # Log-probs under the frozen policy — the denominator of the ratio.",
        "        with torch.no_grad():",
        "            old_logp = torch.distributions.Categorical(logits=policy(obs_batch)).log_prob(act_batch)",
        "        # Reuse the group for several clipped-surrogate update epochs.",
        "        last_loss, last_entropy = 0.0, 0.0",
        f"        for _ in range({update_epochs}):",
        "            dist = torch.distributions.Categorical(logits=policy(obs_batch))",
        "            ratio = torch.exp(dist.log_prob(act_batch) - old_logp)",
        f"            clipped = torch.clamp(ratio, {1 - clip!r}, {1 + clip!r})",
        "            entropy = dist.entropy().mean()",
        f"            loss = -torch.min(ratio * adv, clipped * adv).mean() - {beta!r} * entropy",
        "            opt.zero_grad()",
        "            loss.backward()",
        "            opt.step()",
        "            last_loss, last_entropy = float(loss.item()), float(entropy.item())",
        '        history["mean_return"].append(sum(ep_returns) / len(ep_returns))',
        '        history["episode_len"].append(sum(ep_lens) / len(ep_lens))',
        '        history["policy_loss"].append(last_loss)',
        '        history["entropy"].append(last_entropy)',
        "        if on_epoch is None:",
        f'            print(f"iter {{iteration + 1}}/{epochs}  return {{history[\'mean_return\'][-1]:.1f}}  entropy {{history[\'entropy\'][-1]:.3f}}")',
        "        if on_epoch is not None and on_epoch(iteration + 1, history) is False:",
        "            break",
        "    env.close()",
        "    return history",
    ]
    return "\n".join(lines) + "\n"


GRPO = RecipeDef(
    name="grpo",
    label="GRPO (clipped policy gradient)",
    roles=[RoleDef("policy", "Policy")],
    params=GRPO_PARAMS,
    role_params={},
    needs_targets=False,
    has_val=False,
    data_role="policy",
    generate=_grpo_generate,
    bind=_reinforce_bind,  # single policy role — the same env-only invocation
    data="env",
)


RECIPES: dict[str, RecipeDef] = {
    SUPERVISED.name: SUPERVISED, GAN.name: GAN, CGAN.name: CGAN, VAE.name: VAE,
    REINFORCE.name: REINFORCE, GRPO.name: GRPO,
}

DEFAULT_RECIPE = "supervised"


def get_recipe(name: str | None) -> RecipeDef | None:
    """The recipe by name, defaulting to supervised for an unset name. Returns
    None for an unknown name (the caller surfaces a clear error)."""
    return RECIPES.get(name or DEFAULT_RECIPE)
