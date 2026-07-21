"""Describe data objects (tensors / arrays / Datasets / DataLoaders) so the Data
panel can list them with useful metadata and derive the model's Input shape.

Pure functions over an explicit ``namespace`` dict — in production that's the
session's data registry (see ``datastore``), populated by ``sess.data(...)``;
tests inject plain dicts. No copying anywhere: values are references, read in
the same process (the backend lives inside the kernel).
"""
from __future__ import annotations

from typing import Any


def _describe(name: str, value: Any) -> dict[str, Any] | None:
    """Metadata for a data-like value, or None if it isn't one. Duck-typed and
    fully defensive so one odd variable can't break the listing."""
    # Raw text IS data for a language model: the sequence source tokenizes it
    # (and keeps the vocabulary, so samples can be read back as text). Reported
    # with the two facts that matter — how much text, and how many symbols.
    if isinstance(value, str):
        return {
            "name": name, "kind": "text",
            "num_samples": len(value), "vocab_size": len(set(value)),
        }
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset

        if isinstance(value, torch.Tensor):
            return {
                "name": name, "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
        if isinstance(value, DataLoader):
            n = None
            try:
                n = len(value.dataset)  # type: ignore[arg-type]
            except Exception:
                pass
            return {
                "name": name, "kind": "dataloader",
                "batch_size": value.batch_size, "num_samples": n,
            }
        if isinstance(value, Dataset):
            n = None
            try:
                n = len(value)  # type: ignore[arg-type]
            except Exception:
                pass
            return {"name": name, "kind": "dataset", "num_samples": n}
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {"name": name, "kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return {"name": name, "kind": "dataframe", "shape": [len(value), value.shape[1]]}
    except Exception:
        pass

    return None


def list_data_variables(namespace: dict[str, Any]) -> list[dict[str, Any]]:
    """Data-like entries of a namespace dict, with light metadata. Skips
    private/dunder names."""
    ns = namespace
    out: list[dict[str, Any]] = []
    for name, value in list(ns.items()):
        if name.startswith("_"):
            continue
        try:
            entry = _describe(name, value)
        except Exception:
            entry = None
        if entry is not None:
            out.append(entry)
    return out


def variable_kind(name: str, namespace: dict[str, Any]) -> str | None:
    """The kind ("tensor"/"ndarray"/"dataset"/"dataloader"/"dataframe") of a named
    variable, or None if absent/not data-like. Drives type-aware DataLoader codegen."""
    ns = namespace
    if name not in ns:
        return None
    try:
        d = _describe(name, ns[name])
    except Exception:
        return None
    return d["kind"] if d else None


def _arraylike_spec(x: Any) -> tuple[list[int], bool] | None:
    """(shape, is_integer_dtype) for a torch Tensor or numpy array, else None."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            ints = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
            return list(x.shape), x.dtype in ints
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return list(x.shape), bool(np.issubdtype(x.dtype, np.integer))
    except Exception:
        pass
    return None


def input_shape_for(name: str, namespace: dict[str, Any]) -> dict[str, str] | None:
    """Derive the model's Input shape+dtype from a named variable, so the Data
    panel can populate the Input node. Returns e.g. {"shape": "1, 32",
    "dtype": "long"} (batch dim placeholdered as 1), or None if not inferable.

    A raw batched tensor/array has a leading batch dim (dropped); a Dataset/
    DataLoader yields one un-batched sample, whose first element is the input."""
    value = namespace.get(name)
    if value is None:
        return None

    al = _arraylike_spec(value)
    if al is not None:
        dims, is_int = al[0][1:], al[1]  # drop batch dim
    else:
        try:
            from torch.utils.data import DataLoader, Dataset
        except Exception:
            return None
        dataset = value.dataset if isinstance(value, DataLoader) else value if isinstance(value, Dataset) else None
        if dataset is None:
            return None
        try:
            sample = dataset[0]
        except Exception:
            return None
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        spec = _arraylike_spec(x)
        if spec is None:
            return None
        dims, is_int = spec  # a single sample has no batch dim

    shape = ", ".join(str(int(d)) for d in [1, *dims])  # placeholder batch = 1
    return {"shape": shape, "dtype": "long" if is_int else "float"}
