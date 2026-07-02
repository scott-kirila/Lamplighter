"""Object introspection (metadata + Input-shape derivation over an explicit
namespace dict) and the session data registry that feeds it in production:
sess.data(X=X, y=y) registers references by name; the Data tab lists exactly
those; the runner resolves names at run start."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend import datastore
from backend.introspect import input_shape_for, list_data_variables, variable_kind


@pytest.fixture(autouse=True)
def _clean_registry():
    datastore.clear()
    yield
    datastore.clear()


# --- introspection over an explicit dict -----------------------------------

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


# --- the data registry -------------------------------------------------------

def test_register_merges_across_calls():
    datastore.register(X=torch.randn(4, 2))
    datastore.register(y=torch.randint(0, 3, (4,)))  # a second call adds
    assert set(datastore.registry()) == {"X", "y"}


def test_reregister_repoints_a_name_without_copying():
    a, b = torch.randn(4, 2), torch.randn(8, 2)
    datastore.register(X=a)
    assert datastore.registry()["X"] is a  # a reference — the same object
    datastore.register(X=b)  # re-run-the-cell idiom: repoint the name
    assert datastore.registry()["X"] is b


def test_in_place_mutation_is_visible_through_the_registry():
    X = torch.zeros(3)
    datastore.register(X=X)
    X[0] = 7.0  # mutate in the notebook — no re-registration needed
    assert datastore.registry()["X"][0].item() == 7.0


def test_drop_removes_and_unknown_names_error():
    datastore.register(X=torch.randn(2), y=torch.randn(2))
    datastore.drop("X")
    assert set(datastore.registry()) == {"y"}
    with pytest.raises(ValueError, match="not registered: X"):
        datastore.drop("X")


def test_register_rejects_non_data_objects():
    with pytest.raises(ValueError, match="'lr' is a float"):
        datastore.register(lr=0.001)
    assert datastore.registry() == {}  # nothing partially registered


def test_summary_carries_metadata():
    datastore.register(X=torch.randn(20, 8))
    s = datastore.summary()
    assert s["X"]["kind"] == "tensor" and s["X"]["shape"] == [20, 8]


# --- the Session API (kernel-side wrappers) ----------------------------------

def test_session_data_api():
    from lamplighter import LamplighterError
    from lamplighter.session import Session

    sess = Session("127.0.0.1", 1)  # no server needed — in-process registry
    listing = sess.data(X=torch.randn(10, 4), y=torch.randint(0, 2, (10,)))
    assert set(listing) == {"X", "y"}
    assert sess.list_data()["X"]["shape"] == [10, 4]
    assert set(sess.drop_data("X")) == {"y"}
    with pytest.raises(LamplighterError):
        sess.data(oops="not data")
    assert sess.data() == sess.list_data()  # no-arg call just lists
