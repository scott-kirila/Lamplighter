"""Tier 1 #1 — the declarative-node guardrail.

Every ModuleEmit node must construct via `getattr(nn, cls)` with the args
`build_module_args` produces and run on the meta device. A typo'd class name or a
bad `derived`/param spec fails here, immediately, the moment a node is added.
"""
import pytest
import torch
import torch.nn as nn

from backend.registry import REGISTRY, ModuleEmit, build_module_args

# Input shapes that satisfy each node's needs: rank (conv/pool layers want a
# specific dimensionality) and a batch size > 1 (BatchNorm1d rejects batch size 1
# in training mode). Anything not listed uses the 2D fallback.
INPUT_SHAPE = {
    "Conv1d": [8, 3, 16],
    "Conv2d": [8, 3, 8, 8],
    "Conv3d": [8, 3, 8, 8, 8],
    "MaxPool2d": [8, 3, 8, 8],
    "AvgPool2d": [8, 3, 8, 8],
    "AdaptiveAvgPool2d": [8, 3, 8, 8],
    "GroupNorm": [8, 16, 4, 4],  # 16 channels, divisible by default num_groups=8
}
FALLBACK_SHAPE = [8, 16]

MODULE_NODES = [(name, nd) for name, nd in REGISTRY.items() if isinstance(nd.emit, ModuleEmit)]


@pytest.mark.parametrize("name,node_def", MODULE_NODES, ids=[n for n, _ in MODULE_NODES])
def test_module_emit_builds_and_runs(name, node_def):
    input_shape = INPUT_SHAPE.get(name, FALLBACK_SHAPE)
    pos, kw = build_module_args(node_def, node_def.default_params(), input_shape)
    with torch.device("meta"):
        module = getattr(nn, node_def.emit.cls)(*pos, **kw)
        out = module(torch.empty(input_shape))
    assert len(out.shape) >= 1


def test_registry_covers_expected_kinds():
    # Lock the split so a future miscategorization is visible. Every standard
    # node is a ModuleEmit; only the IO/Concat nodes are bespoke.
    module = {n for n, d in REGISTRY.items() if isinstance(d.emit, ModuleEmit)}
    bespoke = {n for n, d in REGISTRY.items() if d.emit is None}
    assert module == {
        "Linear", "Conv1d", "Conv2d", "Conv3d", "MaxPool2d", "AvgPool2d",
        "AdaptiveAvgPool2d", "Flatten", "Dropout", "BatchNorm1d", "LayerNorm",
        "GroupNorm", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU",
    }
    assert bespoke == {"Input", "Output", "Concat"}
