"""The fidelity gate: can a registry node express this module *exactly*?

A verification tool that draws a wrong picture is worse than one that admits a
gap. So before the importer accepts a module as a typed node, it asks this: does
the node's registry definition cover every constructor argument the module was
built with that differs from torch's default? If not, the module becomes an
``Opaque`` node — a labelled box with a recorded output shape, honest about
being a hole — rather than a confidently-wrong ``Conv2d`` that dropped its
``groups``.

The gate reads the module's live attributes against the nn class's own
constructor signature, which is where the ground truth lives. Two categories
are excused, because a difference in them is not infidelity:

* ``device``/``dtype``/``inplace`` — don't affect structure or weights.
* a difference that torch itself defaults to — a uniform tuple where the
  scalar default lives (``dilation=(1,1)`` vs ``1``) reads as "no change".
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .importer import IGNORED_CTOR_ARGS, _coerce
from .registry import REGISTRY, ModuleEmit


@dataclass
class Verdict:
    opaque: bool
    reason: str = ""


def _expressible_params(node_type: str) -> set[str]:
    """The constructor args a node's registry definition can carry — the union
    of its emit's positional param names and its kwargs. Derived args (a conv's
    in_channels) are excluded on purpose: they come from the wiring."""
    emit = REGISTRY[node_type].emit
    assert isinstance(emit, ModuleEmit)
    names = set(emit.kw_params)
    for arg in emit.pos:
        if isinstance(arg, str):
            names.add(arg)
    return names


def _normalize(value: Any) -> Any:
    """A live module attribute reduced to something comparable to a constructor
    default. A bias-like arg is stored as a Parameter (when on) or None (when
    off), so it maps to the bool the constructor took — and a Parameter can't be
    compared with ``==`` (a multi-element tensor's truth value is ambiguous), so
    it MUST be collapsed before it reaches a comparison."""
    import torch

    if isinstance(value, torch.Tensor):
        return True   # a stored weight/bias means the arg was on
    return value


def _defaults_equal(a: Any, b: Any) -> bool:
    """Whether a live value and a constructor default mean the same layer,
    tolerant of the tuple/scalar spelling torch uses internally (``(3, 3)`` ==
    ``3``) and of the Parameter-or-None encoding of a bool arg (``bias``)."""
    a = _normalize(a)
    if a is None and isinstance(b, bool):
        a = False     # a None attribute where the arg is a bool → the arg was off
    if a == b:
        return True
    ca, cb = _coerce(a, "tuple"), _coerce(b, "tuple")
    return ca == cb


def assess_module(module, node_type: str | None) -> Verdict:
    """Should this module be a typed ``node_type`` node, or Opaque?

    ``node_type`` is None when the class has no registry node at all — an
    automatic Opaque. Otherwise, every constructor arg whose live value differs
    from the class default must be one the node can express; the first that
    isn't sends the whole module Opaque, naming it.
    """
    if node_type is None:
        return Verdict(opaque=True, reason=f"{type(module).__name__} has no canvas node")

    cls = type(module)
    try:
        sig = inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return Verdict(opaque=False)  # can't introspect (rare) — trust the type map

    expressible = _expressible_params(node_type)
    for name, param in sig.items():
        if name in ("self", "args", "kwargs") or name in IGNORED_CTOR_ARGS:
            continue
        if param.default is inspect.Parameter.empty:
            continue  # required arg — comes from wiring/derived, not a fidelity risk
        if not hasattr(module, name):
            continue  # torch didn't keep it as an attribute; can't compare
        actual = getattr(module, name)
        if _defaults_equal(actual, param.default):
            continue  # left at default — nothing to express
        if name not in expressible:
            return Verdict(
                opaque=True,
                reason=f"{cls.__name__}(…{name}={_short(actual)}…) — the {node_type} "
                       f"node can't carry {name}",
            )
    return Verdict(opaque=False)


def _short(value: Any) -> str:
    s = repr(value)
    return s if len(s) <= 24 else s[:21] + "…"
