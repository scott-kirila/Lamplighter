"""Data panel (Slice 1): generate_dataloader() emits a make_dataloaders() helper
from the data config, and it composes with the DataLoader training mode."""
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.codegen import generate_dataloader, generate_module, generate_training
from backend.schema import Graph
from tests.helpers import edge, graph, node


# --- tensors source -------------------------------------------------------

def test_tensors_no_val_returns_single_loader():
    code = generate_dataloader(Graph(data={"source": "tensors", "batch_size": 8}))
    assert "def make_dataloaders(X, y, *, batch_size=8):" in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    train_loader, val_loader = ns["make_dataloaders"](torch.randn(20, 4), torch.randint(0, 3, (20,)))
    assert val_loader is None
    xb, yb = next(iter(train_loader))
    assert xb.shape[0] <= 8


def test_tensors_val_split_partitions_disjointly():
    code = generate_dataloader(Graph(data={"source": "tensors", "val_split": 0.25, "batch_size": 8}))
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
    code = generate_dataloader(Graph(data={
        "source": "torchvision", "dataset": "MNIST", "root": "/data", "download": False}))
    assert "from torchvision import datasets, transforms" in code
    assert "transforms.ToTensor()" in code
    assert "def make_dataloaders(*, batch_size=32, root='/data'):" in code  # root a param
    assert "datasets.MNIST(root, train=True, download=False, transform=transform)" in code
    assert "datasets.MNIST(root, train=False, download=False, transform=transform)" in code


# --- form definition ------------------------------------------------------

def test_data_params_expose_show_if():
    with TestClient(app) as c:
        params = {p["name"]: p for p in c.get("/api/data/params").json()}
    assert params["dataset"]["show_if"] == {"source": "torchvision"}
    assert params["root"]["show_if"] == {"source": "torchvision"}
    assert params["val_split"]["show_if"] == {"source": "tensors"}
    assert params["source"]["show_if"] is None  # always shown
    assert params["batch_size"]["show_if"] is None


# --- integration: Data panel output feeds the Training panel output --------

def test_dataloader_pipeline_end_to_end():
    g = graph(
        [node("in", "Input", {"shape": "8, 64"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    g.data = {"source": "tensors", "val_split": 0.25, "batch_size": 8}
    g.training = {"data": "dataloader", "epochs": 2, "lr": 0.05, "device": "cpu"}

    mns: dict = {}
    exec(generate_module(g), mns)  # noqa: S102
    dns: dict = {}
    exec(generate_dataloader(g), dns)  # noqa: S102
    tns: dict = {}
    exec(generate_training(g), tns)  # noqa: S102

    X, y = torch.randn(24, 64), torch.randint(0, 3, (24,))
    train_loader, val_loader = dns["make_dataloaders"](X, y)
    # make_dataloaders() output flows straight into the generated train().
    tns["train"](mns["GeneratedModel"](), train_loader, val_loader=val_loader)
