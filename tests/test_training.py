"""Training codegen: config → a clean train() function, and the generated loop
actually trains a model."""
import torch
import torch.nn as nn

from backend.codegen import generate_training
from backend.schema import Graph


def _code(training=None):
    return generate_training(Graph(training=training or {}))


def test_default_loss_and_optimizer():
    code = _code()
    assert "nn.CrossEntropyLoss()" in code
    assert "torch.optim.Adam(model.parameters(), lr=0.001)" in code
    assert "weight_decay" not in code  # default 0.0 omitted


def test_custom_loss_optimizer_and_weight_decay():
    code = _code({"loss": "MSELoss", "optimizer": "SGD", "lr": 0.05, "weight_decay": 1e-4})
    assert "nn.MSELoss()" in code
    assert "torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0001)" in code


def test_epochs_and_batch_size_baked_in():
    code = _code({"epochs": 5, "batch_size": 16})
    assert "def train(model, X, y, *, epochs=5, batch_size=16):" in code


def test_generated_train_actually_trains():
    # exec the generated train(), run it on a tiny model, assert the loss drops.
    ns: dict = {}
    exec(_code({"epochs": 40, "batch_size": 8, "lr": 0.05}), ns)  # noqa: S102
    train = ns["train"]

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))

    before = nn.functional.cross_entropy(model(X), y).item()
    returned = train(model, X, y)
    after = nn.functional.cross_entropy(model(X), y).item()

    assert returned is model
    assert after < before
