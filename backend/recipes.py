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
from typing import Callable

from .codegen import generate_training
from .registry import TRAINING_PARAMS, ParamDef
from .schema import Project, graph_from_project


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
    generate: Callable[[Project], str]


def _supervised_generate(project: Project) -> str:
    """The classic loop. The sole model's graph carries the project-level
    training config back onto it, so the emitted source is byte-identical to
    ``generate_training(graph)`` for a single-model project."""
    return generate_training(graph_from_project(project))


SUPERVISED = RecipeDef(
    name="supervised",
    label="Supervised",
    roles=[RoleDef("model", "Model")],
    params=list(TRAINING_PARAMS),
    role_params={},
    needs_targets=True,
    has_val=True,
    generate=_supervised_generate,
)


RECIPES: dict[str, RecipeDef] = {SUPERVISED.name: SUPERVISED}

DEFAULT_RECIPE = "supervised"


def get_recipe(name: str | None) -> RecipeDef | None:
    """The recipe by name, defaulting to supervised for an unset name. Returns
    None for an unknown name (the caller surfaces a clear error)."""
    return RECIPES.get(name or DEFAULT_RECIPE)
