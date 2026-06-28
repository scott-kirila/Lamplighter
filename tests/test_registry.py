"""Tier 2 — the arg builder and the API contract.

`build_module_args` casting/defaults is what keeps codegen output stable as params
are added; `get_registry` must keep `emit` out of the API payload the frontend
consumes.
"""
from backend.app import get_registry
from backend.registry import REGISTRY, build_module_args


def test_conv_args_derive_cast_and_default():
    # in_channels derived from input_shape[1]; out_channels cast from str;
    # kernel_size/stride/padding fall back to defaults.
    pos, kw = build_module_args(REGISTRY["Conv2d"], {"out_channels": "16"}, [1, 3, 8, 8])
    assert pos == [3, 16, 3]
    assert kw == {"stride": 1, "padding": 0, "padding_mode": "zeros"}


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
