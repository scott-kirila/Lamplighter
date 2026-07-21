"""Pretrained backbones: a torchvision model as a feature extractor you draw a
head onto. Shapes come from probing the real architecture (never the weights —
inference must not reach the network), and freezing means what it says."""
import pytest
import torch

from lamplighter.backend.codegen import generate_dataloader, generate_module, generate_training
from lamplighter.backend.diagnose import diagnose
from lamplighter.backend.inference import infer_shapes
from lamplighter.backend.registry import BACKBONES
from lamplighter.backend.schema import Graph
from tests.helpers import edge, graph, node, single_model_project


def _backbone_graph(arch="resnet18", pretrained=False, freeze=True, shape="1, 3, 64, 64", out=3):
    return graph(
        [
            node("in", "Input", {"shape": shape, "dtype": "float"}),
            node("bb", "Backbone", {"arch": arch, "pretrained": pretrained, "freeze": freeze}),
            node("head", "Linear", {"out_features": out}),
            node("out", "Output"),
        ],
        [edge("in", "bb"), edge("bb", "head"), edge("head", "out")],
    )


# --- shape inference ----------------------------------------------------------

def test_every_curated_backbone_yields_a_flat_feature_vector():
    # The curation rule: head-stripping must leave rank-2 features, or the head
    # you draw can't be a Linear. (convnext_tiny fails this and is excluded.)
    for arch, spec in BACKBONES.items():
        g = _backbone_graph(arch=arch, shape="1, 3, 224, 224")
        shapes, errors = infer_shapes(g)
        assert not errors, (arch, errors)
        feats = shapes[("bb", "output")]
        assert len(feats) == 2, (arch, feats)
        # And the spec's documented width is torchvision's actual answer.
        assert feats[1] == spec.features, (arch, feats, spec.features)


def test_the_head_is_sized_from_the_probed_feature_width():
    # The payoff of probing over tabulating: the Linear you draw gets its
    # in_features from the backbone itself, with nothing to keep in sync.
    src = generate_module(_backbone_graph(arch="resnet18", out=10))
    assert "self.layer_1 = nn.Linear(512, 10)" in src
    assert "nn.Linear(2048, 10)" in generate_module(_backbone_graph(arch="resnet50", out=10))


def test_inference_never_asks_for_pretrained_weights(monkeypatch):
    # Shape inference runs on every edit; if it downloaded weights, editing a
    # graph offline would break. Weights don't affect shapes, so it builds the
    # architecture alone — even with pretrained on.
    import torchvision.models as models

    real = models.resnet18
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return real(*args, **kwargs)

    monkeypatch.setattr(models, "resnet18", spy)
    infer_shapes(_backbone_graph(pretrained=True))
    assert seen and all(w is None for w in seen), seen


# --- generated code -----------------------------------------------------------

def test_generated_code_shows_all_three_steps():
    src = generate_module(_backbone_graph(pretrained=True, freeze=True))
    assert "from torchvision import models" in src
    assert "from torchvision.models import ResNet18_Weights" in src
    assert "self.layer_0 = models.resnet18(weights=ResNet18_Weights.DEFAULT)" in src
    assert "self.layer_0.fc = nn.Identity()" in src  # features, not logits
    assert "p.requires_grad = False" in src


def test_an_unfrozen_unpretrained_backbone_generates_the_plain_form():
    src = generate_module(_backbone_graph(pretrained=False, freeze=False))
    assert "models.resnet18(weights=None)" in src
    assert "_Weights" not in src  # no enum import when nothing is pretrained
    assert "requires_grad" not in src
    assert "def train(self" not in src  # nothing frozen → nothing to pin


def test_an_unknown_architecture_is_refused_not_injected():
    # The arch name lands in source as an attribute, so it's validated against
    # the curated table — the torchvision-dataset and loss-name rule.
    g = _backbone_graph(arch="os.system")
    with pytest.raises(ValueError, match="unknown backbone 'os.system'"):
        generate_module(g)


# --- what freezing actually means ---------------------------------------------

def test_freezing_trains_only_the_head_and_pins_batchnorm():
    g = _backbone_graph(freeze=True, out=3)
    ns: dict = {}
    exec(generate_module(g), ns)  # noqa: S102
    model = ns["GeneratedModel"]()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 512 * 3 + 3  # exactly the head

    # requires_grad=False stops the WEIGHTS moving; BatchNorm's running stats
    # would still drift in train mode, so a frozen backbone is pinned to eval.
    bn_before = model.layer_0.bn1.running_mean.clone()
    head_before = model.layer_1.weight.clone()
    tns: dict = {}
    exec(generate_training(g, {"epochs": 1, "device": "cpu", "lr": 0.1, "metric": "none"}), tns)  # noqa: S102
    dns: dict = {}
    exec(generate_dataloader(Graph(), {"source": "memory", "batch_size": 4}), dns)  # noqa: S102
    loader, _ = dns["make_dataloaders"](torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8,)))
    tns["train"](model, loader)

    assert torch.equal(bn_before, model.layer_0.bn1.running_mean)  # pinned
    assert not torch.equal(head_before, model.layer_1.weight)  # the head learned
    assert model.training and not model.layer_0.training  # train() honored the pin


def test_the_optimizer_skips_frozen_weights_only_when_something_is_frozen():
    frozen = generate_training(_backbone_graph(freeze=True), {})
    assert "torch.optim.Adam([p for p in model.parameters() if p.requires_grad]" in frozen
    # Nothing frozen anywhere → the plain line every other model has always had.
    plain = generate_training(_backbone_graph(freeze=False), {})
    assert "torch.optim.Adam(model.parameters()" in plain


# --- diagnostics ---------------------------------------------------------------

def _levels(checks, level):
    return [c for c in checks if c["level"] == level]


def _titles(checks):
    return " | ".join(c["title"] for c in checks)


def test_diagnose_reports_what_a_backbone_expects():
    project = single_model_project(
        _backbone_graph(pretrained=True, shape="1, 3, 224, 224"),
        data={"source": "memory", "x_var": "X", "y_var": "y"},
    )
    ns = {"X": torch.randn(8, 3, 224, 224), "y": torch.randint(0, 3, (8,))}
    rows = diagnose(project, ns)
    oks = _titles(_levels(rows, "ok"))
    assert "resnet18: frozen" in oks
    assert "resnet18 weights download on first use" in oks
    detail = next(c["detail"] for c in rows if c["title"].startswith("resnet18: frozen"))
    assert "512 features" in detail
    # In-memory tensors are standardized in the notebook where they're made,
    # so there's no normalization advice to give here.
    assert not any("normaliz" in c["title"] for c in rows)


def test_a_pretrained_backbone_asks_for_the_statistics_it_learned_with():
    # Feeding pretrained weights differently-scaled inputs isn't an error — it
    # just quietly underperforms, which is exactly what a checklist is for.
    def rows(normalize):
        project = single_model_project(
            _backbone_graph(pretrained=True, shape="1, 3, 224, 224"),
            data={"source": "imagefolder", "root": "./imgs", "resize": 224, "normalize": normalize},
        )
        return diagnose(project, {})

    warns = _titles(_levels(rows("none"), "warn"))
    assert "inputs aren't normalized, but resnet18 was trained on ImageNet-standardized images" in warns
    assert "inputs are ImageNet-normalized" in _titles(_levels(rows("imagenet"), "ok"))
    # An image folder has no statistics of its own to offer.
    assert "an image folder has no known statistics to normalize with" in _titles(
        _levels(rows("dataset"), "error"))

    # A torchvision set does — but they're still the wrong ones for a backbone.
    tv = single_model_project(
        _backbone_graph(pretrained=True, shape="1, 3, 224, 224"),
        data={"source": "torchvision", "dataset": "CIFAR10", "resize": 224, "normalize": "dataset"},
    )
    assert "inputs use this dataset's statistics, but resnet18 was trained on ImageNet's" in _titles(
        _levels(diagnose(tv, {}), "warn"))


def test_an_unpretrained_backbone_gets_no_normalization_advice():
    # Trained from scratch, it learns whatever scaling it's given.
    project = single_model_project(
        _backbone_graph(pretrained=False, shape="1, 3, 224, 224"),
        data={"source": "imagefolder", "root": "./imgs", "resize": 224},
    )
    assert not any("normaliz" in c["title"] for c in diagnose(project, {}))


def test_diagnose_catches_the_channel_and_resolution_traps():
    # Grayscale into an ImageNet backbone is a hard stop...
    grey = single_model_project(_backbone_graph(shape="1, 1, 28, 28"),
                                data={"source": "memory", "x_var": "X", "y_var": "y"})
    ns = {"X": torch.randn(8, 1, 28, 28), "y": torch.randint(0, 3, (8,))}
    errors = _titles(_levels(diagnose(grey, ns), "error"))
    assert "resnet18 needs 3-channel images but gets 1" in errors

    # ...and tiny RGB images are a warning: the features stop being worth much.
    small = single_model_project(_backbone_graph(shape="1, 3, 32, 32"),
                                 data={"source": "memory", "x_var": "X", "y_var": "y"})
    ns = {"X": torch.randn(8, 3, 32, 32), "y": torch.randint(0, 3, (8,))}
    warns = _titles(_levels(diagnose(small, ns), "warn"))
    assert "resnet18 sees 32×32 images" in warns


def test_a_fine_tuning_backbone_says_so():
    project = single_model_project(_backbone_graph(freeze=False, shape="1, 3, 224, 224"),
                                   data={"source": "memory", "x_var": "X", "y_var": "y"})
    ns = {"X": torch.randn(8, 3, 224, 224), "y": torch.randint(0, 3, (8,))}
    assert "resnet18: fine-tuning all weights" in _titles(diagnose(project, ns))


# --- the health panel sees it as a layer ---------------------------------------

def test_a_backbone_counts_as_a_layer_for_telemetry():
    from lamplighter.backend.codegen import layer_nodes

    layers = {ln.layer: ln for ln in layer_nodes(_backbone_graph())}
    assert layers["layer_0"].type == "Backbone"
    assert layers["layer_1"].type == "Linear"


def test_backbone_param_counts_ride_inference():
    counts: dict = {}
    infer_shapes(_backbone_graph(), param_counts=counts)
    assert counts["bb"]["count"] > 10_000_000  # resnet18 ≈ 11.2M
