"""Introspection spike: prove the backend can read the notebook's live variables
— including from a background thread, since the real server runs in a uvicorn
thread inside the kernel. If this holds, the Data panel's notebook-variable
source is feasible.

Uses a plain InteractiveShell to stand in for the Jupyter kernel; the
get_ipython() singleton + user_ns access is the same mechanism as the real one.
"""
import threading

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.introspect import (
    input_shape_for,
    list_data_variables,
    user_namespace,
    variable_kind,
)


# --- namespace-injected unit tests (no IPython) ---------------------------

def test_filters_to_data_like_values():
    ns = {
        "X": torch.randn(20, 8),
        "y": torch.randint(0, 3, (20,)),
        "loader": DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4)), batch_size=2),
        "ds": TensorDataset(torch.randn(4, 2), torch.zeros(4)),
        "lr": 0.001,                 # scalar — not data
        "make": lambda z: z,          # function — not data
        "_hidden": torch.randn(2),    # private — skipped
    }
    found = {v["name"]: v for v in list_data_variables(ns)}
    assert set(found) == {"X", "y", "loader", "ds"}
    assert found["X"]["kind"] == "tensor" and found["X"]["shape"] == [20, 8]
    assert found["y"]["dtype"] == "int64"
    assert found["loader"]["kind"] == "dataloader" and found["loader"]["batch_size"] == 2
    assert found["ds"]["kind"] == "dataset" and found["ds"]["num_samples"] == 4


def test_no_kernel_degrades_gracefully():
    # Outside IPython, user_namespace() must not raise (it may return __main__).
    assert isinstance(user_namespace(), dict)


def test_variable_kind():
    ns = {
        "X": torch.randn(2, 2),
        "dl": DataLoader(TensorDataset(torch.randn(2, 2), torch.zeros(2)), batch_size=1),
        "ds": TensorDataset(torch.randn(2, 2), torch.zeros(2)),
    }
    assert variable_kind("X", ns) == "tensor"
    assert variable_kind("dl", ns) == "dataloader"
    assert variable_kind("ds", ns) == "dataset"
    assert variable_kind("missing", ns) is None


def test_input_shape_from_tensor_drops_batch_dim():
    ns = {"X": torch.randn(20, 784), "idx": torch.randint(0, 5, (20,))}
    assert input_shape_for("X", ns) == {"shape": "1, 784", "dtype": "float"}
    assert input_shape_for("idx", ns)["dtype"] == "long"  # integer tensor


def test_input_shape_from_image_tensor():
    ns = {"imgs": torch.randn(8, 1, 28, 28)}
    assert input_shape_for("imgs", ns) == {"shape": "1, 1, 28, 28", "dtype": "float"}


def test_input_shape_from_dataset_sample():
    # A Dataset yields one un-batched (x, y); the Input shape comes from x.
    ns = {"ds": TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,)))}
    assert input_shape_for("ds", ns) == {"shape": "1, 3, 32, 32", "dtype": "float"}


def test_input_shape_missing_var_is_none():
    assert input_shape_for("nope", {}) is None


# --- the spike: read the real IPython namespace from a background thread ---

def test_reads_ipython_namespace_from_a_thread():
    IPython = pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()  # the get_ipython() singleton
    try:
        # Define variables the way a notebook cell would.
        shell.run_cell(
            "import torch\n"
            "features = torch.randn(16, 32)\n"
            "labels = torch.randint(0, 5, (16,))\n"
            "learning_rate = 0.01\n"
        )
        assert IPython.get_ipython() is shell  # singleton reachable

        # Introspect from a *separate thread* — the uvicorn server scenario.
        result: dict = {}
        t = threading.Thread(
            target=lambda: result.update(
                {v["name"]: v for v in list_data_variables()}  # no namespace arg -> live user_ns
            )
        )
        t.start()
        t.join()

        assert "features" in result and result["features"]["shape"] == [16, 32]
        assert "labels" in result
        assert "learning_rate" not in result  # scalar filtered out
    finally:
        InteractiveShell.clear_instance()
