"""Introspect the notebook's live variables so the Data panel can offer the
user's actual data objects (tensors / arrays / Datasets / DataLoaders).

This works because the Lamplighter backend runs *inside the Jupyter kernel*
(a daemon thread in the same process), so it can read the IPython interactive
namespace directly — no data leaves the kernel. Everything degrades gracefully
(returns empty / None) when there's no kernel, so it's safe under tests and a
bare server.
"""
from __future__ import annotations

from typing import Any


def user_namespace() -> dict[str, Any]:
    """The notebook's variable namespace, or {} when not in IPython.

    Prefers the IPython interactive shell (the canonical place cell variables
    live); falls back to __main__ globals, and to {} if neither is usable.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except Exception:
        pass
    try:
        import __main__

        return vars(__main__)
    except Exception:
        return {}


def _describe(name: str, value: Any) -> dict[str, Any] | None:
    """Metadata for a data-like value, or None if it isn't one. Duck-typed and
    fully defensive so one odd variable can't break the listing."""
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


def list_data_variables(namespace: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Data-like variables in the notebook namespace, with light metadata.
    Skips private/dunder names. `namespace` is injectable for tests."""
    ns = user_namespace() if namespace is None else namespace
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
