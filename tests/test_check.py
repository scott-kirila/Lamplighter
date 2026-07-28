"""The headless check engine: ``lamplighter.check(model, data, loss=...)``.

Each test plants one of the silent-failure classes the engine exists for and
asserts it is caught — and, just as important, that a clean setup comes back
green and that findings are *reported* rows, never raised exceptions. The
model side is always a real ``nn.Module`` (no graph), because that is the
engine's whole claim.
"""
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

from lamplighter.backend.checks import CheckReport, check


def _mlp(in_features=8, out_features=3, bn=False):
    layers = [nn.Linear(in_features, 16)]
    if bn:
        layers.append(nn.BatchNorm1d(16))
    layers += [nn.ReLU(), nn.Linear(16, out_features)]
    return nn.Sequential(*layers)


def _data(n=30, feats=8, classes=3):
    torch.manual_seed(0)
    return torch.randn(n, feats), torch.randint(0, classes, (n,))


def _titles(report, level=None):
    rows = report.rows if level is None else [r for r in report.rows if r["level"] == level]
    return " | ".join(r["title"] for r in rows)


# --- the clean path ----------------------------------------------------------

def test_clean_setup_is_green():
    X, y = _data()
    report = check(_mlp(), (X, y), loss=nn.CrossEntropyLoss())
    assert report.ok, _titles(report)
    assert not report.warnings, _titles(report, "warn")
    assert "forward pass" in _titles(report)
    assert "aligned" in _titles(report)
    assert "classes 0…2 match" in _titles(report)


def test_report_shapes():
    X, y = _data()
    report = check(_mlp(), (X, y), loss=nn.CrossEntropyLoss())
    text = repr(report)
    assert text.startswith("✓") and "✓ forward pass" in text
    d = report.to_dict()
    assert d["ok"] is True and d["errors"] == 0
    assert all({"level", "title", "detail"} <= set(r) for r in d["checks"])
    assert len(report) == len(list(report))


# --- the flagship: labels vs output width ------------------------------------

def test_label_off_by_one_is_an_error():
    X, _ = _data()
    y = torch.randint(1, 4, (30,))  # 1…3 into a 3-logit head — silent on MPS
    report = check(_mlp(out_features=3), (X, y), loss=nn.CrossEntropyLoss())
    assert not report.ok
    assert "has classes 1…3 but the model outputs 3" in _titles(report, "error")


def test_range_check_runs_without_a_loss():
    # Integer 1-D labels against (N, C) outputs is classification in any loop —
    # a missing loss= must not silence the one check that matters most.
    X, _ = _data()
    y = torch.randint(1, 4, (30,))
    report = check(_mlp(out_features=3), (X, y))
    assert "has classes 1…3" in _titles(report, "error")
    assert "no loss given" in _titles(report, "warn")


def test_unused_head_width_warns():
    X, y = _data(classes=3)
    report = check(_mlp(out_features=10), (X, y), loss=nn.CrossEntropyLoss())
    assert "the model outputs 10 classes but 'y' only uses 0…2" in _titles(report, "warn")


# --- loss ↔ target dtype/shape ----------------------------------------------

def test_float_labels_under_ce():
    X, y = _data()
    report = check(_mlp(), (X, y.float()), loss=nn.CrossEntropyLoss())
    assert "needs integer class targets but 'y' is float" in _titles(report, "error")
    detail = next(r["detail"] for r in report.errors)
    assert "y.long()" in detail  # headless fix, not sess.data(...)


def test_column_vector_target():
    X, y = _data()
    report = check(_mlp(), (X, y.unsqueeze(1)), loss=nn.CrossEntropyLoss())
    assert "expects 1-D class targets but 'y' is (30, 1)" in _titles(report, "error")


def test_integer_targets_under_mse():
    X, y = _data()
    model = _mlp(out_features=1)
    report = check(model, (X, y), loss=nn.MSELoss())
    assert "MSELoss needs float targets but 'y' is integer" in _titles(report, "error")


def test_functional_loss_is_recognised():
    X, y = _data()
    report = check(_mlp(), (X, y.float()), loss=F.cross_entropy)
    assert "CrossEntropyLoss needs integer class targets" in _titles(report, "error")


def test_unknown_loss_is_not_guessed():
    def contrastive(out, y):  # pragma: no cover - never called
        return out.sum()

    X, y = _data()
    report = check(_mlp(), (X, y), loss=contrastive)
    assert "isn't a torch built-in" in _titles(report)
    assert report.ok  # no dtype rule invented for it


# --- the behavioural activation probe ----------------------------------------

class _SoftmaxInForward(nn.Module):
    """The case no module walk can see: a functional softmax in forward()."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


def test_functional_softmax_under_ce_is_caught():
    X, y = _data()
    report = check(_SoftmaxInForward(), (X, y), loss=nn.CrossEntropyLoss())
    assert "outputs probabilities but CrossEntropyLoss expects raw logits" \
        in _titles(report, "error")


def test_softmax_module_under_ce_is_caught():
    X, y = _data()
    model = nn.Sequential(nn.Linear(8, 3), nn.Softmax(dim=-1))
    report = check(model, (X, y), loss=nn.CrossEntropyLoss())
    assert "outputs probabilities but CrossEntropyLoss" in _titles(report, "error")


def test_logsoftmax_under_ce_warns_redundant():
    X, y = _data()
    model = nn.Sequential(nn.Linear(8, 3), nn.LogSoftmax(dim=-1))
    report = check(model, (X, y), loss=nn.CrossEntropyLoss())
    assert report.ok  # harmless, so not an error
    assert "log-probabilities under CrossEntropyLoss" in _titles(report, "warn")


def test_nll_pairings():
    X, y = _data()
    good = nn.Sequential(nn.Linear(8, 3), nn.LogSoftmax(dim=-1))
    assert "NLLLoss: matched" in _titles(check(good, (X, y), loss=nn.NLLLoss()))

    bad = nn.Sequential(nn.Linear(8, 3), nn.Softmax(dim=-1))
    report = check(bad, (X, y), loss=nn.NLLLoss())
    assert "NLLLoss expects log-probabilities" in _titles(report, "error")

    raw = check(_mlp(), (X, y), loss=nn.NLLLoss())
    assert "doesn't output log-probabilities" in _titles(raw, "warn")


def test_raw_logits_under_ce_reported_ok():
    X, y = _data()
    report = check(_mlp(), (X, y), loss=nn.CrossEntropyLoss())
    assert "look like raw logits" in _titles(report, "ok")


# --- the forward probe itself -------------------------------------------------

def test_forward_crash_is_a_row_not_an_exception():
    X, y = _data(feats=8)
    report = check(_mlp(in_features=16), (X, y), loss=nn.CrossEntropyLoss())
    assert "the model can't consume a real batch" in _titles(report, "error")


def test_nan_output_is_an_error():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(x) / 0.0

    X, y = _data()
    report = check(Broken(), (X, y), loss=nn.CrossEntropyLoss())
    assert "NaN/Inf" in _titles(report, "error")


def test_batch_dim_folding_is_caught():
    class Folds(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):  # x is (B, 2, 4); view eats the batch dim
            return self.fc(x.reshape(-1, 4))

    X = torch.randn(30, 2, 4)
    report = check(Folds(), (X, torch.randint(0, 3, (30,))), loss=nn.CrossEntropyLoss())
    assert "came out as" in _titles(report, "error")


def test_seq_first_output_warns():
    class SeqFirst(nn.Module):
        def forward(self, x):  # transposes samples and positions
            return x.transpose(0, 1)

    X = torch.randn(8, 20, 4)  # batch-first in, (20, 8, 4) out
    report = check(SeqFirst(), X)
    assert "batch dim second, not first" in _titles(report, "warn")


def test_training_mode_is_restored_and_probe_runs_in_eval():
    seen = {}

    class Records(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            seen["training"] = self.training
            return self.fc(x)

    model = Records().train()
    check(model, _data()[0])
    assert seen["training"] is False  # probed in eval mode
    assert model.training is True  # and put back


# --- data forms ---------------------------------------------------------------

def test_misaligned_x_y():
    X, y = _data(n=30)
    report = check(_mlp(), (X, y[:29]), loss=nn.CrossEntropyLoss())
    assert "'X' has 30 samples but 'y' has 29" in _titles(report, "error")


def test_numpy_is_flagged_but_still_checked():
    X, y = _data()
    report = check(_mlp(), (X.numpy(), torch.randint(1, 4, (30,)).numpy()),
                   loss=nn.CrossEntropyLoss())
    assert "'X' is a numpy array" in _titles(report, "error")
    # The conversion is flagged, not fatal — the range check still ran:
    assert "has classes 1…3" in _titles(report, "error")


def test_dataloader_labels_are_checked():
    # The hero regression: wrapping the same data in a DataLoader must not
    # silence the class-range check (the silent-skip family's worst member).
    X, _ = _data()
    y = torch.randint(1, 4, (30,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    report = check(_mlp(out_features=3), loader, loss=nn.CrossEntropyLoss())
    assert "has classes 1…3 but the model outputs 3" in _titles(report, "error")


def test_custom_dataset_falls_back_to_first_batch_scope():
    class Pairs(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    X, y = _data()
    report = check(_mlp(), DataLoader(Pairs(X, y), batch_size=8),
                   loss=nn.CrossEntropyLoss())
    row = next(r for r in report.rows if "first-batch labels" in r["title"])
    assert row["level"] == "ok" and "only one batch was read" in row["detail"]

    bad = torch.randint(5, 9, (30,))
    report = check(_mlp(), DataLoader(Pairs(X, bad), batch_size=8),
                   loss=nn.CrossEntropyLoss())
    assert "first-batch labels run 5…8 but the model outputs 3" in _titles(report, "error")


def test_unlabelled_data_with_a_loss_says_so():
    class Unlabelled(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return torch.randn(8)

    report = check(_mlp(), Unlabelled(), loss=nn.CrossEntropyLoss())
    assert "no labels found" in _titles(report, "warn")


def test_bare_dataset_is_probed_via_collate():
    X, y = _data()
    report = check(_mlp(), TensorDataset(X, y), loss=nn.CrossEntropyLoss())
    assert "forward pass" in _titles(report)
    assert "classes 0…2 match" in _titles(report)


def test_iterable_dataset_consumption_is_announced():
    class Stream(IterableDataset):
        def __iter__(self):
            return iter([(torch.randn(8), torch.tensor(0)) for _ in range(16)])

    report = check(_mlp(), DataLoader(Stream(), batch_size=4),
                   loss=nn.CrossEntropyLoss())
    assert "IterableDataset" in _titles(report, "warn")


def test_hf_style_dict_batch():
    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, inputs, labels=None):
            return {"logits": self.fc(inputs), "loss": None}

    data = {"inputs": torch.randn(30, 8), "labels": torch.randint(1, 4, (30,))}
    report = check(Wrapped(), data, loss=nn.CrossEntropyLoss())
    assert "forward pass" in _titles(report)  # dict fed as kwargs, logits unwrapped
    assert "has classes 1…3" in _titles(report, "error")


def test_empty_data_is_an_error():
    report = check(_mlp(), (torch.zeros(0, 8), torch.zeros(0, dtype=torch.long)))
    assert "0 samples" in _titles(report, "error")


# --- sequence-shaped outputs --------------------------------------------------

def test_seq_output_class_range_without_the_1d_rule():
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 16)
            self.head = nn.Linear(16, 10)

        def forward(self, x):
            return self.head(self.emb(x))

    X = torch.randint(0, 10, (4, 12))
    y_ok = torch.randint(0, 10, (4, 12))
    report = check(TinyLM(), (X, y_ok), loss=nn.CrossEntropyLoss())
    # (B, T) integer targets must NOT trip the 1-D rule for (B, T, C) outputs
    assert "expects 1-D class targets" not in _titles(report)
    assert "classes" in _titles(report, "ok")

    y_over = torch.randint(0, 12, (4, 12))
    report = check(TinyLM(), (X, y_over), loss=nn.CrossEntropyLoss())
    assert "but the model outputs 10" in _titles(report, "error")


# --- imbalance ----------------------------------------------------------------

def test_imbalance_warns_with_real_counts():
    X = torch.randn(44, 8)
    y = torch.tensor([0] * 40 + [1] * 4)
    report = check(_mlp(out_features=2), (X, y), loss=nn.CrossEntropyLoss())
    row = next(r for r in report.warnings if "imbalanced" in r["title"])
    assert "10:1" in row["title"]
    assert "0: 40, 1: 4" in row["detail"]
    assert "weight= on CrossEntropyLoss" in row["detail"]


# --- batch arithmetic ---------------------------------------------------------

def test_batchnorm_ragged_final_batch_from_loader():
    X, y = _data(n=33)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)  # 33 % 8 = 1
    report = check(_mlp(bn=True), loader, loss=nn.CrossEntropyLoss())
    assert "the final batch has 1 sample" in _titles(report, "error")

    fixed = DataLoader(TensorDataset(X, y), batch_size=8, drop_last=True)
    assert check(_mlp(bn=True), fixed, loss=nn.CrossEntropyLoss()).ok


def test_batchnorm_with_batch_size_param():
    X, y = _data()
    report = check(_mlp(bn=True), (X, y), loss=nn.CrossEntropyLoss(), batch_size=1)
    assert "batch_size 1 with BatchNorm1d" in _titles(report, "error")


def test_batchnorm_without_batch_info_says_unchecked():
    X, y = _data()
    report = check(_mlp(bn=True), (X, y), loss=nn.CrossEntropyLoss())
    assert "no batch size to check it against" in _titles(report, "warn")


def test_drop_last_waste_warns():
    X, y = _data(n=40)
    loader = DataLoader(TensorDataset(X, y), batch_size=25, drop_last=True)  # 15/40 dropped
    report = check(_mlp(), loader, loss=nn.CrossEntropyLoss())
    assert "discards 15 of 40 samples" in _titles(report, "warn")


# --- misuse is an exception, findings are rows --------------------------------

def test_api_misuse_raises():
    X, y = _data()
    with pytest.raises(TypeError):
        check("not a module", (X, y))
    with pytest.raises(TypeError):
        check(_mlp(), None)
    with pytest.raises(TypeError):
        check(_mlp(), (X, y, y))
    with pytest.raises(TypeError):
        check(_mlp(), (X, y), y)  # targets twice


def test_report_type():
    X, y = _data()
    assert isinstance(check(_mlp(), (X, y), loss="CrossEntropyLoss"), CheckReport)
