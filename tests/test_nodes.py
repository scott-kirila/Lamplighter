"""Tier 1 #1 — the declarative-node guardrail.

Every ModuleEmit node must construct via `getattr(nn, cls)` with the args
`build_module_args` produces and run on the meta device. A typo'd class name or a
bad `derived`/param spec fails here, immediately, the moment a node is added.
"""
import pytest
import torch
import torch.nn as nn

from backend.registry import REGISTRY, ModuleEmit, build_module_args

# Input shapes that satisfy each node's needs: rank (Conv2d/MaxPool2d want 4D)
# and a batch size > 1 (BatchNorm1d rejects batch size 1 in training mode).
INPUT_SHAPE = {"Conv2d": [8, 3, 8, 8], "MaxPool2d": [8, 3, 8, 8]}
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
    # Lock the split so a future miscategorization is visible.
    from backend.registry import FunctionalEmit

    module = {n for n, d in REGISTRY.items() if isinstance(d.emit, ModuleEmit)}
    functional = {n for n, d in REGISTRY.items() if isinstance(d.emit, FunctionalEmit)}
    bespoke = {n for n, d in REGISTRY.items() if d.emit is None}
    assert module == {"Linear", "Conv2d", "MaxPool2d", "Flatten", "Dropout", "BatchNorm1d"}
    assert functional == {"ReLU", "Sigmoid", "Tanh"}
    assert bespoke == {"Input", "Output", "Concat"}
