"""Data panel: generate_dataloader() emits a make_dataloaders() helper from the
data config — the single data path feeding train(model, loader, val_loader)."""
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from lamplighter.backend.app import app
from lamplighter.backend.codegen import generate_dataloader, generate_module, generate_training
from lamplighter.backend.schema import Graph
from tests.helpers import edge, graph, node, single_model_project


# --- memory source: generic tensors (no pick) ----------------------------

def test_tensors_no_val_returns_single_loader():
    code = generate_dataloader(Graph(), {"source": "memory", "batch_size": 8})
    assert "def make_dataloaders(X, y, *, batch_size=8):" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    train_loader, val_loader = ns["make_dataloaders"](torch.randn(20, 4), torch.randint(0, 3, (20,)))
    assert val_loader is None
    xb, yb = next(iter(train_loader))
    assert xb.shape[0] <= 8


def test_drop_last_applies_to_train_loader_only():
    off = generate_dataloader(Graph(), {"source": "memory"})
    assert "drop_last" not in off  # omitted when off, for clean code
    on = generate_dataloader(Graph(), {"source": "memory", "val_split": 0.2, "drop_last": True})
    assert "shuffle=True, drop_last=True)" in on  # train loader
    assert "val_loader = DataLoader(val_ds, batch_size=batch_size)" in on  # val untouched


def test_tensors_val_split_partitions_disjointly():
    code = generate_dataloader(Graph(), {"source": "memory", "val_split": 0.25, "batch_size": 8})
    assert "random_split" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    tl, vl = ns["make_dataloaders"](torch.randn(20, 4), torch.randint(0, 3, (20,)))
    assert vl is not None
    # 20 samples, 25% held out -> 15 train / 5 val, disjoint.
    assert sum(yb.size(0) for _, yb in tl) == 15
    assert sum(yb.size(0) for _, yb in vl) == 5


# --- torchvision source (string-only; no dataset download) ----------------

def test_torchvision_mnist_codegen():
    code = generate_dataloader(Graph(), {
        "source": "torchvision", "dataset": "MNIST", "root": "/data", "download": False})
    assert "from torchvision import datasets, transforms" in code
    assert "transforms.ToTensor()" in code
    assert "def make_dataloaders(*, batch_size=32, root='/data'):" in code  # root a param
    assert "datasets.MNIST(root, train=True, download=False, transform=transform)" in code
    assert "datasets.MNIST(root, train=False, download=False, transform=transform)" in code


# --- memory source: picked variable (type-aware wrapping) ---------------

def test_variable_source_dataloader_passes_through():
    dl = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4)), batch_size=2)
    code = generate_dataloader(
        Graph(), {"source": "memory", "x_var": "loader"}, namespace={"loader": dl})
    assert "def make_dataloaders(loader):" in code
    assert "return loader, None" in code  # already a DataLoader — nothing to build


def test_variable_source_dataset_gets_wrapped():
    ds = TensorDataset(torch.randn(4, 2), torch.zeros(4))
    code = generate_dataloader(
        Graph(), {"source": "memory", "x_var": "ds", "batch_size": 16}, namespace={"ds": ds})
    assert "def make_dataloaders(dataset, *, batch_size=16):" in code
    assert "DataLoader(dataset, batch_size=batch_size, shuffle=True)" in code


def test_variable_source_tensor_falls_back_to_tensordataset():
    ns = {"X": torch.randn(20, 8), "y": torch.randint(0, 3, (20,))}
    code = generate_dataloader(Graph(), {"source": "memory", "x_var": "X"}, namespace=ns)
    assert "TensorDataset(X, y)" in code  # tensor pick → the X,y wrapping


# --- Slice 3: datasets, augmentations, perf knobs -------------------------

def test_more_torchvision_datasets():
    code = generate_dataloader(Graph(), {"source": "torchvision", "dataset": "CIFAR10"})
    assert "datasets.CIFAR10(" in code


def test_augmentations_are_train_only_in_canonical_order():
    code = generate_dataloader(Graph(), {
        "source": "torchvision", "augmentations": ["Grayscale", "RandomHorizontalFlip"]})
    # Canonical order (flip before grayscale) regardless of selection order, ToTensor last.
    assert ("train_transform = transforms.Compose(["
            "transforms.RandomHorizontalFlip(), transforms.Grayscale(), transforms.ToTensor()])") in code
    assert "eval_transform = transforms.Compose([transforms.ToTensor()])" in code


def test_randomcrop_auto_sizes_to_the_dataset_and_leads_the_augmentations():
    # CIFAR: 32px crop with the standard padding=4, first in the train chain,
    # never in eval (it's augmentation).
    code = generate_dataloader(Graph(), {
        "source": "torchvision", "dataset": "CIFAR10",
        "augmentations": ["RandomHorizontalFlip", "RandomCrop"]})
    assert "transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()" in code
    assert "eval_transform = transforms.Compose([transforms.ToTensor()])" in code
    # MNIST auto-sizes to 28; a resize overrides the crop size.
    assert "transforms.RandomCrop(28, padding=4)" in generate_dataloader(
        Graph(), {"source": "torchvision", "dataset": "MNIST", "augmentations": ["RandomCrop"]})
    assert "transforms.RandomCrop(64, padding=4)" in generate_dataloader(
        Graph(), {"source": "torchvision", "dataset": "MNIST", "resize": 64, "augmentations": ["RandomCrop"]})


def test_normalize_uses_canonical_stats_on_both_transforms():
    # Preprocessing, not augmentation: after ToTensor on train AND eval, with
    # the dataset's own stats.
    code = generate_dataloader(Graph(), {
        "source": "torchvision", "dataset": "CIFAR10",
        "augmentations": ["RandomHorizontalFlip"], "normalize": True})
    stat = "transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))"
    assert code.count(stat) == 2  # train + eval
    assert f"eval_transform = transforms.Compose([transforms.ToTensor(), {stat}])" in code
    mnist = generate_dataloader(Graph(), {"source": "torchvision", "dataset": "MNIST", "normalize": True})
    # No augmentations → one shared transform, still normalized.
    assert "transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])" in mnist
    assert "transform=train_transform" in code and "transform=eval_transform" in code
    compile(code, "<gen>", "exec")  # generated code parses


def test_no_augmentations_uses_one_shared_transform():
    code = generate_dataloader(Graph(), {"source": "torchvision"})
    assert "train_transform" not in code
    assert "    transform = transforms.Compose([transforms.ToTensor()])" in code


def test_perf_knobs_apply_to_all_loaders_when_set():
    off = generate_dataloader(Graph(), {"source": "memory"})
    assert "num_workers" not in off and "pin_memory" not in off
    on = generate_dataloader(Graph(), {"source": "memory", "num_workers": 4, "pin_memory": True})
    assert on.count("num_workers=4, pin_memory=True") == 1  # single (no-val) train loader


def test_slice3_params_shape():
    with TestClient(app) as c:
        params = {p["name"]: p for p in c.get("/api/data/params").json()}
    assert params["augmentations"]["type"] == "multienum"
    assert "RandomHorizontalFlip" in params["augmentations"]["choices"]
    assert params["num_workers"]["show_if"] == {"advanced": True}
    assert params["pin_memory"]["show_if"] == {"advanced": True}


# --- Slice 3 remainder: ImageFolder + Resize ------------------------------

def test_imagefolder_with_val_split():
    code = generate_dataloader(Graph(), {
        "source": "imagefolder", "root": "./imgs", "resize": 224, "val_split": 0.2})
    assert "datasets.ImageFolder(root, transform=transform)" in code
    assert "transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])" in code
    # The split is carved with a fixed generator, stable across runs/resumes.
    assert "split = torch.Generator().manual_seed(1234)" in code
    assert "random_split(dataset, [n_train, n_val], generator=split)" in code
    assert "def make_dataloaders(*, batch_size=32, root='./imgs', val_split=0.2):" in code
    compile(code, "<gen>", "exec")


def test_imagefolder_without_val_split_single_loader():
    code = generate_dataloader(Graph(), {"source": "imagefolder", "root": "./imgs"})
    assert "random_split" not in code
    assert "return train_loader, None" in code


def test_val_split_holds_the_same_samples_across_run_seeds():
    # The held-out set must not depend on the training seed — so a resume, which
    # draws a fresh seed, validates on exactly the same samples (comparable metrics).
    import torch

    code = generate_dataloader(Graph(), {"source": "memory", "x_var": "X", "y_var": "y", "val_split": 0.25})
    ns: dict = {}
    exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102
    make = ns["make_dataloaders"]

    X = torch.arange(40).float().unsqueeze(1)  # row i carries label i, so the split is identifiable
    y = torch.arange(40)

    def held_out(seed):
        torch.manual_seed(seed)  # stands in for two different run seeds
        _, val_loader = make(X, y, batch_size=8)
        return sorted(int(row[1]) for row in val_loader.dataset)

    a, b = held_out(0), held_out(999)
    assert a == b  # same held-out samples regardless of the seed
    assert len(a) == 10  # 25% of 40 — an actual split happened


def test_resize_leads_both_transforms_in_torchvision():
    code = generate_dataloader(Graph(), {
        "source": "torchvision", "resize": 32, "augmentations": ["RandomHorizontalFlip"]})
    assert "train_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])" in code
    assert "eval_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])" in code


def test_resize_off_when_unset():
    code = generate_dataloader(Graph(), {"source": "torchvision"})
    assert "Resize" not in code


def test_show_if_lists_for_shared_fields():
    with TestClient(app) as c:
        params = {p["name"]: p for p in c.get("/api/data/params").json()}
    assert params["root"]["show_if"] == {"source": ["torchvision", "imagefolder"]}
    assert params["resize"]["show_if"] == {"source": ["torchvision", "imagefolder"]}
    assert params["val_split"]["show_if"] == {"source": ["memory", "imagefolder"]}
    assert "imagefolder" in params["source"]["choices"]


# --- form definition ------------------------------------------------------

def test_data_params_expose_show_if():
    with TestClient(app) as c:
        params = {p["name"]: p for p in c.get("/api/data/params").json()}
    assert params["dataset"]["show_if"] == {"source": "torchvision"}  # single-value rule
    assert params["source"]["show_if"] is None  # always shown
    assert params["batch_size"]["show_if"] is None
    # (root/resize/val_split use list-valued rules — see test_show_if_lists_for_shared_fields)


# --- data picker endpoint (the session registry) ---------------------------

def test_data_variables_endpoint_lists_registered_data_with_shapes():
    from lamplighter.backend import datastore

    try:
        datastore.register(feats=torch.randn(12, 20))
        with TestClient(app) as c:
            variables = {v["name"]: v for v in c.get("/api/data/variables").json()["variables"]}
        assert set(variables) == {"feats"}  # exactly what's registered — no scan
        # The endpoint enriches each entry with the Input shape it implies.
        assert variables["feats"]["input_shape"] == {"shape": "1, 20", "dtype": "float"}
    finally:
        datastore.clear()


# --- multi-input models ---------------------------------------------------

def _two_input_graph():
    return graph(
        [
            node("a", "Input", {"shape": "4, 8"}, y=0),
            node("b", "Input", {"shape": "4, 8"}, y=100),
            node("cat", "Concat", {"dim": 1}),
            node("lin", "Linear", {"out_features": 3}),
            node("out", "Output"),
        ],
        [edge("a", "cat", tgt_h="in0"), edge("b", "cat", tgt_h="in1"),
         edge("cat", "lin"), edge("lin", "out")],
    )


def test_multi_input_tensors_one_x_per_input():
    g = _two_input_graph()
    code = generate_dataloader(g, {"source": "memory", "batch_size": 8})
    assert "def make_dataloaders(X0, X1, y, *, batch_size=8):" in code
    assert "TensorDataset(X0, X1, y)" in code


def test_post_data_code_reflects_posted_graph_input_count():
    # The Data tab POSTs the live project so the preview matches the canvas without
    # depending on backend-state sync (fixes reload staleness).
    project = single_model_project(_two_input_graph(), data={"source": "memory", "batch_size": 8})
    with TestClient(app) as c:
        code = c.post("/api/data/code", json=project.model_dump()).json()["code"]
    assert "def make_dataloaders(X0, X1, y" in code  # two inputs from the posted graph


def test_multi_input_dataloader_pipeline_end_to_end():
    g = _two_input_graph()
    dns: dict = {}
    exec(generate_dataloader(g, {"source": "memory", "val_split": 0.25, "batch_size": 8}), dns)  # noqa: S102
    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    tns: dict = {}
    exec(generate_training(g, {"epochs": 1, "device": "cpu"}), tns)  # noqa: S102
    X0, X1, y = torch.randn(24, 8), torch.randn(24, 8), torch.randint(0, 3, (24,))
    train_loader, val_loader = dns["make_dataloaders"](X0, X1, y)  # X per input
    tns["train"](mns["GeneratedModel"](), train_loader, val_loader=val_loader)  # *xb, yb


# --- integration: Data panel output feeds the Training panel output --------

def test_dataloader_pipeline_end_to_end():
    g = graph(
        [node("in", "Input", {"shape": "8, 64"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    dns: dict = {}
    exec(generate_dataloader(g, {"source": "memory", "val_split": 0.25, "batch_size": 8}), dns)  # noqa: S102
    tns: dict = {}
    exec(generate_training(g, {"epochs": 2, "lr": 0.05, "device": "cpu"}), tns)  # noqa: S102

    X, y = torch.randn(24, 64), torch.randint(0, 3, (24,))
    train_loader, val_loader = dns["make_dataloaders"](X, y)
    # make_dataloaders() output flows straight into the generated train().
    tns["train"](mns["GeneratedModel"](), train_loader, val_loader=val_loader)
