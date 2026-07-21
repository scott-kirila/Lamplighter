"""Data↔model diagnostics: the Data tab's pre-run checklist. Because the
registry holds real references, these checks read actual shapes/dtypes/values —
catching mismatches (including the class-range crash) before a run starts."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from lamplighter.backend.diagnose import diagnose
from tests.helpers import edge, graph, node, single_model_project


def _mlp(input_shape="1, 8", out_features=3, name="", loss=None, data=None):
    g = graph(
        [
            node("in", "Input", {"shape": input_shape, "dtype": "float", "name": name}),
            node("l", "Linear", {"out_features": out_features}),
            node("out", "Output"),
        ],
        [edge("in", "l"), edge("l", "out")],
    )
    return single_model_project(
        g,
        training={"loss": loss} if loss else {},
        data={"source": "memory", "x_var": "X", "y_var": "y", **(data or {})},
    )


def _ns(n=20, feats=8, classes=3):
    torch.manual_seed(0)
    return {"X": torch.randn(n, feats), "y": torch.randint(0, classes, (n,))}


def _levels(checks, level):
    return [c for c in checks if c["level"] == level]


def _titles(checks):
    return " | ".join(c["title"] for c in checks)


# --- GAN (project-aware) contract -------------------------------------------

def _gan_project(disc_in="1, 8"):
    from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project

    gen = graph(
        [node("in", "Input", {"shape": "1, 16"}), node("l", "Linear", {"out_features": 8}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    disc = graph(
        [node("in", "Input", {"shape": disc_in}), node("l", "Linear", {"out_features": 1}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return Project(
        models=[
            ModelDef(id="g", name="Generator", graph=Graph(nodes=gen.nodes, edges=gen.edges)),
            ModelDef(id="d", name="Discriminator", graph=Graph(nodes=disc.nodes, edges=disc.edges)),
        ],
        data_nodes=[DataNode(id="real", kind="dataset", name="Data",
                             config={"source": "memory", "x_var": "X", "batch_size": 8})],
        links=[ModelLink(id="L", source_data="real", target_model="d")],
        training={"recipe": "gan", "roles": {"generator": "g", "discriminator": "d"}},
    )


def test_gan_checks_the_discriminator_and_needs_no_target():
    checks = diagnose(_gan_project(disc_in="1, 8"), namespace={"X": torch.randn(20, 8)})
    # X (8-dim) matches the discriminator's Input — checked against the data-fed
    # model, not the generator (whose Input is the 16-dim latent).
    assert _levels(checks, "error") == [], _titles(checks)
    assert any("No target needed" in c["title"] for c in checks)
    # No spurious "Target: nothing picked" (needs_targets is False).
    assert not any("Target" in c["title"] and "picked" in c["title"] for c in checks)


def test_cgan_labels_are_conditioning_not_class_targets():
    # The cGAN's y rides the loader as a conditioning label, not a supervised
    # target — the recipe bakes its BCE loss into the loop (no loss knob), so
    # the classification target↔loss fit must not run: the discriminator
    # outputs 1 logit while y spans 0…9, which the loss-fit check would (and
    # once did) flag as an error, making the template unrunnable from the app.
    from lamplighter.backend.templates import TEMPLATES

    project = TEMPLATES["cgan"].build()
    data = next(dn for dn in project.data_nodes if dn.kind == "dataset")
    data.config.update({"x_vars": {"main": "X", "label": "y"}, "y_var": "y"})
    ns = {"X": torch.randn(64, 784), "y": torch.randint(0, 10, (64,))}
    checks = diagnose(project, ns)
    assert _levels(checks, "error") == [], _titles(checks)
    assert any("aligned" in c["title"] for c in checks)


def test_gan_flags_x_not_matching_the_discriminator_input():
    checks = diagnose(_gan_project(disc_in="1, 784"), namespace={"X": torch.randn(20, 8)})
    # X is 8-dim but the discriminator expects 784 — a real mismatch surfaces.
    assert any(c["level"] == "error" and "≠ Input" in c["title"] for c in checks), _titles(checks)


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
    project = _mlp()
    project.models[0].graph.nodes[0].params["dtype"] = "long"  # Input expects indices, X is float
    errs = _titles(_levels(diagnose(project, _ns()), "error"))
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
    project = single_model_project(g, data={"source": "memory", "x_vars": {"a": "X0", "b": "X1"}, "y_var": "y"})
    ns = {"X0": torch.randn(20, 8), "X1": torch.randn(15, 8), "y": torch.randint(0, 3, (20,))}
    errs = _titles(_levels(diagnose(project, ns), "error"))
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
    return single_model_project(g, data={"source": "memory", "x_var": "X", "y_var": "y", **(data or {})})


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
    project = _mlp(input_shape="1, 784", data={"source": "torchvision", "dataset": "MNIST"})
    errs = _titles(_levels(diagnose(project, {}), "error"))
    assert "MNIST yields (1, 28, 28) per sample but the Input is (784)" in errs


def test_imagefolder_without_resize_warns():
    project = _mlp(data={"source": "imagefolder", "root": "./imgs"})
    warns = _titles(_levels(diagnose(project, {}), "warn"))
    assert "vary in size" in warns


# --- broken model still gets data-only checks ------------------------------------

def test_broken_model_reports_but_data_checks_run():
    project = _mlp()
    inner = project.models[0].graph
    inner.edges = inner.edges[:1]  # Output unwired → codegen precondition fails
    ns = _ns()
    ns["y"] = torch.randint(0, 3, (7,))  # count mismatch should still surface
    checks = diagnose(project, ns)
    t = _titles(_levels(checks, "error"))
    assert "Model isn't ready" in t
    assert "'X' has 20 samples but 'y' has 7" in t


# --- logits vs. probabilities (softmax ↔ loss fit) -------------------------------

def _act_mlp(activation, loss, out_features=3):
    """A classifier that optionally ends in `activation` before the Output, with
    the chosen loss — the double-softmax / NLLLoss-head test bed."""
    nodes = [
        node("in", "Input", {"shape": "1, 8", "dtype": "float"}),
        node("l", "Linear", {"out_features": out_features}),
    ]
    edges = [edge("in", "l")]
    last = "l"
    if activation:
        params = {"dim": -1} if activation in ("Softmax", "LogSoftmax") else {}
        nodes.append(node("act", activation, params))
        edges.append(edge("l", "act"))
        last = "act"
    nodes.append(node("out", "Output"))
    edges.append(edge(last, "out"))
    return single_model_project(
        graph(nodes, edges),
        training={"loss": loss},
        data={"source": "memory", "x_var": "X", "y_var": "y"},
    )


def test_softmax_before_crossentropy_is_double_softmax():
    checks = diagnose(_act_mlp("Softmax", "CrossEntropyLoss"), _ns(n=40))
    errs = _levels(checks, "error")
    assert any("CrossEntropyLoss expects raw logits but the model ends in Softmax" in c["title"] for c in errs)
    assert any("applies log-softmax internally" in c["detail"] for c in errs)


def test_logsoftmax_before_crossentropy_is_flagged_too():
    errs = _titles(_levels(diagnose(_act_mlp("LogSoftmax", "CrossEntropyLoss"), _ns(n=40)), "error"))
    assert "CrossEntropyLoss expects raw logits but the model ends in LogSoftmax" in errs


def test_raw_logits_into_crossentropy_is_clean():
    # The correct pairing: no head, raw logits. No logits/probabilities complaint.
    checks = diagnose(_act_mlp(None, "CrossEntropyLoss"), _ns(n=40))
    assert not any("logits" in c["title"] for c in checks)
    assert _levels(checks, "error") == [] and _levels(checks, "warn") == []


def test_softmax_before_nllloss_wants_logsoftmax():
    checks = diagnose(_act_mlp("Softmax", "NLLLoss"), _ns(n=40))
    errs = _levels(checks, "error")
    assert any("NLLLoss expects log-probabilities but the model ends in Softmax" in c["title"] for c in errs)
    assert any("use a LogSoftmax head" in c["detail"] for c in errs)


def test_logsoftmax_before_nllloss_is_the_correct_pairing():
    checks = diagnose(_act_mlp("LogSoftmax", "NLLLoss"), _ns(n=40))
    assert _levels(checks, "error") == []
    assert any("LogSoftmax → NLLLoss" in c["title"] for c in checks)


def test_nllloss_without_a_logsoftmax_head_warns():
    checks = diagnose(_act_mlp(None, "NLLLoss"), _ns(n=40))
    warns = _levels(checks, "warn")
    assert any("NLLLoss expects log-probabilities but the model ends in Linear" in c["title"] for c in warns)
    assert any("add a LogSoftmax" in c["detail"] for c in warns)


def test_activation_loss_check_skips_adversarial_recipes():
    # A GAN bakes its own BCE loss (no loss knob) — the CE/NLL head check must not
    # fire on a discriminator, whatever it ends in.
    checks = diagnose(_gan_project(disc_in="1, 8"), namespace={"X": torch.randn(20, 8)})
    assert not any("logits" in c["title"] or "log-probabilities" in c["title"] for c in checks)


# --- recurrent batch_first vs the batch-first pipeline ---------------------------

def test_drop_last_that_empties_the_train_loader_is_an_error():
    # 10 samples, batch 32, drop_last → every batch is ragged, so every batch is
    # dropped: the loader yields nothing and the loop divides by zero. Fully
    # predictable from n/batch/drop_last, so it must not read as a mere warn.
    project = _mlp(data={"batch_size": 32, "drop_last": True})
    errors = _titles(_levels(diagnose(project, _ns(n=10)), "error"))
    assert "Drop Last with batch_size 32 > 10 training samples leaves no batches" in errors
    # Without drop_last the same numbers are just a single-batch epoch (a warn).
    warns = _titles(_levels(diagnose(_mlp(data={"batch_size": 32}), _ns(n=10)), "warn"))
    assert "batch_size 32 exceeds the 10 training samples" in warns


def test_custom_loss_skips_target_fit_with_a_note():
    # A registered loss class defines its own target contract — no dtype rule
    # applies. Silence would read as "checked and fine"; say what's knowable.
    project = _mlp(loss="Custom")
    rows = diagnose(project, _ns())
    assert "custom loss — target fit isn't checked" in _titles(rows)
    # An integer target under a custom loss is NOT reported as a float-target
    # error (the built-in regression rule must not leak onto it).
    assert not any("needs float targets" in c["title"] for c in rows)


def test_two_datasets_wired_into_one_model_is_an_error():
    # The resolver is first-wire-wins; a silently-losing second dataset must be
    # flagged instead of letting link order decide the run's data.
    from lamplighter.backend.schema import DataNode, ModelLink

    project = _mlp()
    project.data_nodes.append(
        DataNode(id="data2", kind="dataset", name="Data 2", config={"source": "memory"})
    )
    project.links.append(ModelLink(id="L2", source_data="data2", target_model="model"))
    errors = _levels(diagnose(project, _ns()), "error")
    row = next(c for c in errors if "dataset nodes wired" in c["title"])
    assert "2 dataset nodes wired into Model" == row["title"]
    assert "only 'Data'" in row["detail"] and "'Data 2'" in row["detail"]
    # One wired dataset stays quiet.
    assert not any("nodes wired" in c["title"] for c in diagnose(_mlp(), _ns()))


def test_imagefolder_val_split_range_is_checked():
    # The tree's size is unknowable pre-run, so the batching math can't be
    # predicted — but the split range can (and codegen refuses the same rule).
    project = _mlp(data={"source": "imagefolder", "root": "./imgs", "resize": 8, "val_split": 1.0})
    errors = _titles(_levels(diagnose(project, {}), "error"))
    assert "val_split 1.0 — must be in [0, 1)" in errors
    ok = _mlp(data={"source": "imagefolder", "root": "./imgs", "resize": 8, "val_split": 0.2})
    assert "must be in" not in _titles(_levels(diagnose(ok, {}), "error"))


def test_seq_first_recurrent_warns():
    g = graph(
        [
            node("in", "Input", {"shape": "1, 5, 16", "dtype": "float"}),
            node("lstm", "LSTM", {"hidden_size": 8, "batch_first": False}),
            node("l", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("in", "lstm"), edge("lstm", "l", src_h="output"), edge("l", "out")],
    )
    project = single_model_project(g, data={"source": "memory", "x_var": "X", "y_var": "y"})
    torch.manual_seed(0)
    ns = {"X": torch.randn(20, 5, 16), "y": torch.randint(0, 3, (20,))}
    warns = _titles(_levels(diagnose(project, ns), "warn"))
    assert "LSTM has batch_first=False but the pipeline feeds batch-first batches" in warns
    # The default (batch_first=True) is quiet.
    project.models[0].graph.nodes[1].params["batch_first"] = True
    warns = _titles(_levels(diagnose(project, ns), "warn"))
    assert "batch_first" not in warns


# --- class imbalance: the detector and its two remedies -----------------------

def _skewed_ns(n0=90, n1=9, n2=1, feats=8):
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1), torch.full((n2,), 2)]).long()
    return {"X": torch.randn(len(y), feats), "y": y}


def test_imbalance_is_reported_with_its_real_spread():
    rows = diagnose(_mlp(), _skewed_ns())
    warn = next(c for c in _levels(rows, "warn") if "imbalanced" in c["title"])
    assert warn["title"] == "classes are imbalanced (90:1)"
    assert "0: 90, 1: 9, 2: 1" in warn["detail"]  # the counts, not just a verdict
    assert "Class Weights" in warn["detail"] and "Weighted Sampler" in warn["detail"]
    # A balanced set says nothing at all.
    balanced = {"X": torch.randn(30, 8), "y": torch.arange(30) % 3}
    assert not any("imbalanced" in c["title"] for c in diagnose(_mlp(), balanced))


def test_either_remedy_turns_the_imbalance_warning_into_a_confirmation():
    weighted = _mlp()
    weighted.training["class_weights"] = True
    row = next(c for c in diagnose(weighted, _skewed_ns()) if "imbalanced" in c["title"])
    assert row["level"] == "ok" and "class weights rebalances them" in row["title"]

    sampled = _mlp(data={"weighted_sampler": True})
    row = next(c for c in diagnose(sampled, _skewed_ns()) if "imbalanced" in c["title"])
    assert row["level"] == "ok" and "a weighted sampler rebalances them" in row["title"]


def test_both_remedies_at_once_is_flagged_as_double_compensation():
    project = _mlp(data={"weighted_sampler": True})
    project.training["class_weights"] = True
    warns = _titles(_levels(diagnose(project, _skewed_ns()), "warn"))
    assert "class weights AND a weighted sampler are both on" in warns


def test_the_sampler_needs_class_labels():
    # A regression target has no classes to balance by — bincount would bucket
    # continuous values into nonsense, so refuse instead.
    project = _mlp(loss="MSELoss", data={"weighted_sampler": True})
    ns = {"X": torch.randn(20, 8), "y": torch.randn(20, 3)}
    assert "the weighted sampler needs class labels" in _titles(_levels(diagnose(project, ns), "error"))
    # A float 0/1 target (what BCEWithLogits takes) IS balanceable.
    bce = _mlp(out_features=1, loss="BCEWithLogitsLoss", data={"weighted_sampler": True})
    ns01 = {"X": torch.randn(20, 8), "y": torch.cat([torch.zeros(18), torch.ones(2)]).unsqueeze(1)}
    assert not any("needs class labels" in c["title"] for c in diagnose(bce, ns01))


def test_the_sampler_says_when_it_cannot_apply():
    # A picked DataLoader owns sampling; a non-memory source never reaches the
    # sampler path — both would otherwise leave the toggle looking active.
    loader = DataLoader(TensorDataset(torch.randn(20, 8), torch.randint(0, 3, (20,))), batch_size=4)
    picked = _mlp(data={"x_var": "dl", "weighted_sampler": True})
    warns = _titles(_levels(diagnose(picked, {"dl": loader, **_skewed_ns()}), "warn"))
    assert "the picked DataLoader owns its own sampling" in warns

    tv = _mlp(data={"source": "torchvision", "dataset": "MNIST", "weighted_sampler": True})
    warns = _titles(_levels(diagnose(tv, {}), "warn"))
    assert "the weighted sampler doesn't apply to the torchvision source" in warns


# --- the test split -----------------------------------------------------------

def test_test_split_arithmetic_is_reported_and_bounded():
    rows = diagnose(_mlp(data={"val_split": 0.2, "test_split": 0.2}), _ns(n=100))
    oks = _titles(_levels(rows, "ok"))
    assert "test split holds out 20 of 100 samples" in oks
    assert "val split holds out 20 of 100 samples" in oks

    # A fraction that rounds to nothing can't be evaluated on.
    warns = _titles(_levels(diagnose(_mlp(data={"test_split": 0.05}), _ns(n=10)), "warn"))
    assert "test_split 0.05 of 10 samples holds out 0" in warns

    # The two splits are checked TOGETHER — they carve from the same data.
    errors = _titles(_levels(diagnose(_mlp(data={"val_split": 0.6, "test_split": 0.5}), _ns()), "error"))
    assert "val_split 0.6 + test_split 0.5 leaves nothing to train on" in errors
    errors = _titles(_levels(diagnose(_mlp(data={"test_split": 1.5}), _ns()), "error"))
    assert "test_split 1.5 — must be in [0, 1)" in errors


def test_imagefolder_test_split_range_is_checked_too():
    project = _mlp(data={"source": "imagefolder", "root": "./imgs", "resize": 8,
                         "val_split": 0.5, "test_split": 0.6})
    errors = _titles(_levels(diagnose(project, {}), "error"))
    assert "val_split 0.5 + test_split 0.6 leaves nothing to train on" in errors
