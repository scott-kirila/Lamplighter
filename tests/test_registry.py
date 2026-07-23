"""Tier 2 — the arg builder and the API contract.

`build_module_args` casting/defaults is what keeps codegen output stable as params
are added; `get_registry` must keep `emit` out of the API payload the frontend
consumes.
"""
import inspect

import pytest
import torch.nn as nn

from lamplighter.backend.app import get_registry
from lamplighter.backend.registry import (
    REGISTRY,
    ModuleEmit,
    build_module_args,
    render_module_args,
)


def test_conv_args_derive_cast_and_default():
    # in_channels derived from input_shape[1]; out_channels cast from str;
    # kernel_size/stride/padding fall back to defaults. The import-fidelity
    # params (dilation/groups/bias) are built at their torch defaults too — they
    # instantiate the same module, and the render path below omits them.
    pos, kw = build_module_args(REGISTRY["Conv2d"], {"out_channels": "16"}, [1, 3, 8, 8])
    assert pos == [3, 16, 3]
    assert kw == {"stride": 1, "padding": 0, "dilation": 1, "groups": 1,
                  "bias": True, "padding_mode": "zeros"}


def test_new_conv_params_render_nothing_at_their_defaults():
    """The whole reason the fidelity params are safe to add: render_module_args
    omits any kwarg equal to its default, so a hand-built Conv2d emits exactly
    what it always did. A non-default value DOES render."""
    plain = render_module_args(REGISTRY["Conv2d"], {"out_channels": 16}, [1, 3, 8, 8])
    assert plain == "3, 16, 3"  # byte-identical to before the params existed

    depthwise = render_module_args(
        REGISTRY["Conv2d"], {"out_channels": 3, "groups": 3, "bias": False}, [1, 3, 8, 8]
    )
    assert "groups=3" in depthwise and "bias=False" in depthwise


# The load-bearing correctness property for the whole importer: a ParamDef
# whose default disagrees with torch's own constructor default and isn't marked
# always_emit would make render_module_args OMIT a value that torch does NOT
# default to — generating code that diverges from what was imported. This
# asserts every ModuleEmit kwarg either matches torch's default or opts into
# always_emit. (registry.py:32 documents the trap; batch_first is the sanctioned
# exception and is marked always_emit.)
@pytest.mark.parametrize(
    "node_type",
    [k for k, d in REGISTRY.items()
     if isinstance(d.emit, ModuleEmit) and hasattr(nn, d.emit.cls)],
)
def test_module_kwarg_defaults_match_torch_or_always_emit(node_type):
    node = REGISTRY[node_type]
    sig = inspect.signature(getattr(nn, node.emit.cls).__init__).parameters
    pdefs = {p.name: p for p in node.params}
    for name in node.emit.kw_params:
        pd = pdefs[name]
        if name not in sig or sig[name].default is inspect.Parameter.empty:
            continue  # not a simple keyword with a default (positional-only, etc.)
        torch_default = sig[name].default
        if pd.always_emit:
            continue  # deliberately emits regardless — the sanctioned escape
        if callable(torch_default):
            # torch's default is a function (e.g. activation=F.relu); the
            # ParamDef uses the string form torch accepts as equivalent
            # ('relu'), which can't and shouldn't == the function object.
            continue
        assert pd.default == torch_default, (
            f"{node_type}.{name}: ParamDef default {pd.default!r} != torch default "
            f"{torch_default!r} and not always_emit — render would silently diverge"
        )


def test_linear_bias_defaults_to_bool_true():
    pos, kw = build_module_args(REGISTRY["Linear"], {}, [1, 784])
    assert pos == [784, 128]
    assert kw == {"bias": True}


def test_dropout_float_cast():
    _, kw = build_module_args(REGISTRY["Dropout"], {"p": "0.25"}, [1, 16])
    assert kw == {"p": 0.25}


def test_get_registry_strips_emit():
    reg = get_registry()
    assert reg, "registry should not be empty"
    assert all("emit" not in entry for entry in reg.values())
    # presentational fields the frontend relies on are still present
    assert reg["Linear"]["params"]
    assert reg["Input"]["outputs"]
    # subcategory (palette sub-grouping) survives into the payload
    assert reg["Conv2d"]["subcategory"] == "Convolution"


def test_every_layer_has_a_subcategory():
    """The palette groups the (large) Layers category into sub-headers from
    NodeDef.subcategory (see _LAYER_SUBGROUPS). A new layer left out of that map
    would silently render flat — assert none slip through, so the grouping can't
    drift as layers are added."""
    ungrouped = [t for t, d in REGISTRY.items() if d.category == "layers" and not d.subcategory]
    assert not ungrouped, f"layers missing a subcategory (add them to _LAYER_SUBGROUPS): {ungrouped}"


def test_registry_docs_come_live_from_torch():
    """Every node ships help text: nn-backed nodes pull the installed torch's
    docstring (summary = first prose paragraph, reST roles stripped); the
    Lamplighter-native trio (Input/Output/Concat) use their authored line."""
    reg = get_registry()
    assert all(entry["doc"] and entry["doc"]["summary"] for entry in reg.values())

    conv = reg["Conv2d"]["doc"]
    assert conv["summary"].startswith("Applies a 2D convolution")
    assert ":math:" not in conv["body"] and ":class:" not in conv["body"]
    assert "Args:" in conv["body"]  # the fuller text survives for the Inspector

    # LSTM's docstring leads with an __init__ signature — the summary skips it.
    assert reg["LSTM"]["doc"]["summary"].startswith("Apply a multi-layer")

    # Authored, not from torch (these aren't nn modules).
    assert "forward()" in reg["Input"]["doc"]["summary"]
    assert reg["Concat"]["doc"]["summary"].startswith("Concatenates")
