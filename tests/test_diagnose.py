"""Data↔model diagnostics: the Data tab's pre-run checklist. Because the
registry holds real references, these checks read actual shapes/dtypes/values —
catching mismatches (including the class-range crash) before a run starts."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.diagnose import diagnose
from tests.helpers import edge, graph, node


def _mlp(input_shape="1, 8", out_features=3, name="", loss=None, data=None):
    g = graph(
        [
            node("in", "Input", {"shape": input_shape, "dtype": "float", "name": name}),
            node("l", "Linear", {"out_features": out_features}),
            node("out", "Output"),
        ],
        [edge("in", "l"), edge("l", "out")],
    )
    g.training = {"loss": loss} if loss else {}
    g.data = {"source": "memory", "x_var": "X", "y_var": "y", **(data or {})}
    return g


def _ns(n=20, feats=8, classes=3):
    torch.manual_seed(0)
    return {"X": torch.randn(n, feats), "y": torch.randint(0, classes, (n,))}


def _levels(checks, level):
    return [c for c in checks if c["level"] == level]


def _titles(checks):
    return " | ".join(c["title"] for c in checks)


# --- happy path -------------------------------------------------------------

def test_all_ok_for_matching_data():
    checks = diagnose(_mlp(), _ns(n=40))  # > default batch_size, so truly clean
    assert _levels(checks, "error") == [] and _levels(checks, "warn") == []
    t = _titles(checks)
    assert "'X' — 40 samples of (8) match the Input" in t  # batch-dim reading spelled out
    assert "40 samples — 'X' and 'y' aligned" in t
    assert "classes 0…2 match the model's 3 outputs" in t


# --- shape / dtype / pick problems -------------------------------------------

def test_shape_mismatch_flagged():
    checks = diagnose(_mlp(input_shape="1, 100"), _ns(feats=8))
    errs = _titles(_levels(checks, "error"))
    assert "'X' sample (8) ≠ Input (100)" in errs


def test_dtype_mismatch_flagged():
    g = _mlp()
    g.nodes[0].params["dtype"] = "long"  # Input expects indices, X is float
    errs = _titles(_levels(diagnose(g, _ns()), "error"))
    assert "'X' is float but the Input expects integer" in errs


def test_input_missing_batch_placeholder_gets_targeted_warning():
    # Input typed as "8" (the per-sample shape) instead of "1, 8": the generic
    # mismatch would say "sample (8) ≠ Input ()" — useless. Must explain the
    # batch-placeholder convention instead.
    checks = diagnose(_mlp(input_shape="8"), _ns())
    warns = _levels(checks, "warn")
    assert any("missing its leading batch placeholder" in c["title"] for c in warns)
    assert any("set it to (1, 8)" in c["detail"] for c in warns)


def test_single_sample_data_gets_targeted_error():
    # Data registered as one sample (8,) instead of a batch (N, 8).
    ns = _ns()
    ns["X"] = torch.randn(8)
    errs = _levels(diagnose(_mlp(), ns), "error")
    assert any("looks like a single sample, not a batch" in c["title"] for c in errs)
    assert any("expected (N, 8)" in c["detail"] for c in errs)


def test_column_vector_targets_get_squeeze_hint():
    # The (N, 1) CSV classic for classification targets.
    ns = _ns()
    ns["y"] = ns["y"].unsqueeze(1)
    checks = diagnose(_mlp(), ns)
    errs = _levels(checks, "error")
    assert any("expects 1-D class targets" in c["title"] for c in errs)
    assert any("y.squeeze(1)" in c["detail"] for c in errs)


def test_unpicked_and_unregistered():
    assert "Input: nothing picked" in _titles(diagnose(_mlp(data={"x_var": ""}), _ns()))
    ns = _ns()
    del ns["X"]
    assert "'X' is not registered" in _titles(diagnose(_mlp(), ns))


def test_ndarray_pick_advises_from_numpy():
    import numpy as np

    ns = _ns()
    ns["X"] = np.zeros((20, 8))
    checks = diagnose(_mlp(), ns)
    assert "torch.from_numpy" in " ".join(c["detail"] for c in checks)


# --- X/y alignment and loss fit ------------------------------------------------

def test_count_mismatch_flagged():
    ns = _ns()
    ns["y"] = torch.randint(0, 3, (7,))
    errs = _titles(_levels(diagnose(_mlp(), ns), "error"))
    assert "'X' has 20 samples but 'y' has 7" in errs


def test_messages_use_registered_names_not_hardcoded_xy():
    # sess.data(images=..., labels=...) must be diagnosed by those names.
    g = _mlp(data={"x_var": "images", "y_var": "labels"})
    torch.manual_seed(0)
    ns = {"images": torch.randn(40, 8), "labels": torch.randint(0, 3, (40,))}
    t = _titles(diagnose(g, ns))
    assert "'images' — 40 samples of (8) match the Input" in t
    assert "40 samples — 'images' and 'labels' aligned" in t
    ns["labels"] = torch.randint(0, 3, (7,))
    errs = _titles(_levels(diagnose(g, ns), "error"))
    assert "'images' has 40 samples but 'labels' has 7" in errs


def test_class_out_of_range_flagged():
    # y contains class 5 but the model only outputs 3 logits — the check that
    # otherwise surfaces as an opaque device-side assert mid-run.
    ns = _ns()
    ns["y"] = torch.randint(0, 6, (20,))
    errs = _titles(_levels(diagnose(_mlp(out_features=3), ns), "error"))
    assert "but the model outputs 3" in errs


def test_wider_model_than_classes_warns():
    # The "forgot to set out_features" case: 10-class targets into a 128-logit
    # model runs fine mechanically, so it must warn (not pass, not error).
    ns = _ns(classes=10)
    checks = diagnose(_mlp(out_features=128), ns)
    assert _levels(checks, "error") == []
    warns = _levels(checks, "warn")
    assert any("outputs 128 classes but 'y' only uses 0…9" in c["title"] for c in warns)
    assert any("out_features=10" in c["detail"] for c in warns)


def test_float_targets_rejected_for_classification():
    ns = _ns()
    ns["y"] = torch.randn(20)
    errs = _titles(_levels(diagnose(_mlp(loss="CrossEntropyLoss"), ns), "error"))
    assert "needs integer class targets" in errs


def test_integer_targets_rejected_for_regression():
    errs = _titles(_levels(diagnose(_mlp(loss="MSELoss"), _ns()), "error"))
    assert "needs float targets" in errs


def test_multi_input_count_mismatch():
    g = graph(
        [
            node("a", "Input", {"shape": "1, 8"}, y=0),
            node("b", "Input", {"shape": "1, 8"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("l", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"),
         edge("cat", "l"), edge("l", "out")],
    )
    g.data = {"source": "memory", "x_vars": {"a": "X0", "b": "X1"}, "y_var": "y"}
    ns = {"X0": torch.randn(20, 8), "X1": torch.randn(15, 8), "y": torch.randint(0, 3, (20,))}
    errs = _titles(_levels(diagnose(g, ns), "error"))
    assert "different sample counts" in errs


# --- batching / split sanity ---------------------------------------------------

def test_batch_and_val_warnings():
    checks = diagnose(_mlp(data={"batch_size": 64, "val_split": 0.01}), _ns(n=20))
    warns = _titles(_levels(checks, "warn"))
    assert "batch_size 64 exceeds the 20 training samples" in warns
    assert "holds out 0" in warns


def test_batch_compares_against_post_split_training_samples():
    # 40 samples with half held out -> only 20 get batched; batch 32 must warn.
    checks = diagnose(_mlp(data={"batch_size": 32, "val_split": 0.5}), _ns(n=40))
    assert "batch_size 32 exceeds the 20 training samples" in _titles(_levels(checks, "warn"))


def test_invalid_batch_and_val_split_error():
    assert "batch_size 0 — must be at least 1" in _titles(
        _levels(diagnose(_mlp(data={"batch_size": 0}), _ns()), "error"))
    assert "val_split 1.0 — must be in [0, 1)" in _titles(
        _levels(diagnose(_mlp(data={"val_split": 1.0}), _ns()), "error"))


def _bn_mlp(data=None):
    g = graph(
        [
            node("in", "Input", {"shape": "1, 8", "dtype": "float"}),
            node("bn", "BatchNorm1d"),
            node("l", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("in", "bn"), edge("bn", "l"), edge("l", "out")],
    )
    g.data = {"source": "memory", "x_var": "X", "y_var": "y", **(data or {})}
    return g


def test_batchnorm_ragged_final_batch_is_predicted():
    # 33 samples, batch 32 -> final batch of 1 -> BatchNorm crashes in training.
    checks = diagnose(_bn_mlp({"batch_size": 32}), _ns(n=33))
    errs = _levels(checks, "error")
    assert any("final batch has 1 sample and the model contains BatchNorm1d" in c["title"] for c in errs)
    assert any("enable Drop Last" in c["detail"] for c in errs)
    # Drop Last resolves it.
    checks = diagnose(_bn_mlp({"batch_size": 32, "drop_last": True}), _ns(n=33))
    assert _levels(checks, "error") == []


def test_batchnorm_with_batch_size_one_errors_regardless():
    checks = diagnose(_bn_mlp({"batch_size": 1, "drop_last": True}), _ns(n=32))
    assert any("batch_size 1 with BatchNorm1d" in c["title"] for c in _levels(checks, "error"))


def test_no_batchnorm_means_ragged_batch_is_fine():
    checks = diagnose(_mlp(data={"batch_size": 32}), _ns(n=33))
    assert _levels(checks, "error") == []


def test_drop_last_discarding_a_big_share_warns():
    # 100 training samples, batch 64, drop_last -> 36 samples never train.
    checks = diagnose(_mlp(data={"batch_size": 64, "drop_last": True}), _ns(n=100))
    warns = _titles(_levels(checks, "warn"))
    assert "Drop Last discards 36 of 100 training samples" in warns


# --- non-tensor picks and other sources ------------------------------------------

def test_dataloader_pick_checks_sample_shape():
    ns = {"loader": DataLoader(TensorDataset(torch.randn(10, 8), torch.zeros(10)), batch_size=4)}
    checks = diagnose(_mlp(data={"x_var": "loader"}), ns)
    assert _levels(checks, "error") == []
    assert "sample (8) matches" in _titles(checks)


def test_torchvision_shape_mismatch():
    g = _mlp(input_shape="1, 784")
    g.data = {"source": "torchvision", "dataset": "MNIST"}
    errs = _titles(_levels(diagnose(g, {}), "error"))
    assert "MNIST yields (1, 28, 28) per sample but the Input is (784)" in errs


def test_imagefolder_without_resize_warns():
    g = _mlp()
    g.data = {"source": "imagefolder", "root": "./imgs"}
    warns = _titles(_levels(diagnose(g, {}), "warn"))
    assert "vary in size" in warns


# --- broken model still gets data-only checks ------------------------------------

def test_broken_model_reports_but_data_checks_run():
    g = _mlp()
    g.edges = g.edges[:1]  # Output unwired → codegen precondition fails
    ns = _ns()
    ns["y"] = torch.randint(0, 3, (7,))  # count mismatch should still surface
    checks = diagnose(g, ns)
    t = _titles(_levels(checks, "error"))
    assert "Model isn't ready" in t
    assert "'X' has 20 samples but 'y' has 7" in t
