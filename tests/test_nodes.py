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
    "MaxPool1d": [8, 16, 32],
    "MaxPool2d": [8, 3, 8, 8],
    "AvgPool2d": [8, 3, 8, 8],
    "AdaptiveAvgPool2d": [8, 3, 8, 8],
    "AdaptiveMaxPool2d": [8, 3, 8, 8],
    "GroupNorm": [8, 16, 4, 4],  # 16 channels, divisible by default num_groups=8
    "BatchNorm2d": [8, 16, 4, 4],
    "InstanceNorm2d": [8, 16, 4, 4],
    "Dropout2d": [8, 16, 4, 4],
    "Embedding": [8, 10],  # index tensor (the guardrail probes it on the meta device)
    "RNN": [5, 8, 16],  # (seq, batch, features) — recurrent layers want 3D
    "LSTM": [5, 8, 16],
    "GRU": [5, 8, 16],
}
FALLBACK_SHAPE = [8, 16]

MODULE_NODES = [(name, nd) for name, nd in REGISTRY.items() if isinstance(nd.emit, ModuleEmit)]


@pytest.mark.parametrize("name,node_def", MODULE_NODES, ids=[n for n, _ in MODULE_NODES])
def test_module_emit_builds_and_runs(name, node_def):
    input_shape = INPUT_SHAPE.get(name, FALLBACK_SHAPE)
    pos, kw = build_module_args(node_def, node_def.default_params(), input_shape)
    with torch.device("meta"):
        module = getattr(nn, node_def.emit.cls)(*pos, **kw)
        ret = module(torch.empty(input_shape))
    # Navigate each declared output pin (handles multi-output layers like LSTM).
    for _pin, path in node_def.emit.outputs:
        t = ret
        for i in path:
            t = t[i]
        assert len(t.shape) >= 1


def test_registry_covers_expected_kinds():
    # Lock the split so a future miscategorization is visible. Every standard
    # node is a ModuleEmit; only the IO/Concat nodes are bespoke.
    module = {n for n, d in REGISTRY.items() if isinstance(d.emit, ModuleEmit)}
    bespoke = {n for n, d in REGISTRY.items() if d.emit is None}
    assert module == {
        "Linear", "Embedding", "Conv1d", "Conv2d", "Conv3d", "MaxPool1d",
        "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
        "Flatten", "Dropout", "Dropout2d", "BatchNorm1d", "BatchNorm2d",
        "LayerNorm", "GroupNorm", "InstanceNorm2d", "RNN", "LSTM", "GRU",
        "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU", "ELU", "SiLU", "Softmax",
    }
    assert bespoke == {"Input", "Output", "Concat"}
