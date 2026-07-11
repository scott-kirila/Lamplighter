"""Built-in project templates — the "New from template" starting points.

Each template is a complete, working :class:`Project` (models laid out on the
canvas, data wiring provisioned, training config set) built here so the test
suite can hold every one of them green: shapes must infer without errors and
every model must generate code. A template that drifts when the registry
changes fails CI instead of greeting a user with a broken canvas.

Curated for the learning path, roughly in order of ambition: an MLP, a CNN, a
transformer classifier, a GAN, and a VAE — each mirrors the architecture its
example notebook walks through, arriving pre-wired (dataset nodes included,
unconfigured: the user's data names aren't knowable) so a learner can inspect
a working design instead of building from blank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schema import DataNode, Graph, ModelDef, ModelLink, NodePosition, Project

# One canvas column, matching the frontend's insert pitch (node width + gap).
_X = 230


def _n(nid: str, ntype: str, params: dict | None = None, col: int = 0, y: float = 0.0) -> dict:
    return {"id": nid, "type": ntype, "position": {"x": col * _X, "y": y}, "params": params or {}}


def _e(src: str, tgt: str, src_h: str = "output", tgt_h: str = "input") -> dict:
    return {"id": f"{src}->{tgt}:{tgt_h}", "source": src, "sourceHandle": src_h,
            "target": tgt, "targetHandle": tgt_h}


def _chain(nodes: list[dict]) -> Graph:
    """A left-to-right pipeline: each node wired to the next."""
    edges = [_e(a["id"], b["id"]) for a, b in zip(nodes, nodes[1:])]
    return Graph.model_validate({"nodes": nodes, "edges": edges})


def _dataset(target_model: str, *, y: float = 160.0) -> tuple[DataNode, ModelLink]:
    """An unconfigured dataset node wired into a model — what ensureDatasetFor
    would provision; the user picks their registered tensors on it."""
    dn = DataNode(id="data", kind="dataset", name="Data",
                  sys_position=NodePosition(x=-260, y=y), config={"source": "memory"})
    return dn, ModelLink(id="data-link", source_data="data", target_model=target_model)


def _mlp() -> Project:
    graph = _chain([
        _n("in", "Input", {"shape": "1, 784"}, 0),
        _n("l1", "Linear", {"out_features": 128}, 1),
        _n("r1", "ReLU", {}, 2),
        _n("l2", "Linear", {"out_features": 10}, 3),
        _n("out", "Output", {}, 4),
    ])
    dn, link = _dataset("model", y=0.0)
    return Project(
        models=[ModelDef(id="model", name="Model", graph=graph)],
        data_nodes=[dn], links=[link],
        training={"loss": "CrossEntropyLoss", "epochs": 10},
    )


def _cnn() -> Project:
    graph = _chain([
        _n("in", "Input", {"shape": "1, 1, 28, 28"}, 0),
        _n("c1", "Conv2d", {"out_channels": 32, "kernel_size": 3}, 1),
        _n("r1", "ReLU", {}, 2),
        _n("p1", "MaxPool2d", {"kernel_size": 2}, 3),
        _n("c2", "Conv2d", {"out_channels": 64, "kernel_size": 3}, 4),
        _n("r2", "ReLU", {}, 5),
        _n("p2", "MaxPool2d", {"kernel_size": 2}, 6),
        _n("f", "Flatten", {}, 7),
        _n("l", "Linear", {"out_features": 10}, 8),
        _n("out", "Output", {}, 9),
    ])
    dn, link = _dataset("model", y=0.0)
    return Project(
        models=[ModelDef(id="model", name="Model", graph=graph)],
        data_nodes=[dn], links=[link],
        training={"loss": "CrossEntropyLoss", "epochs": 10},
    )


def _transformer() -> Project:
    graph = _chain([
        _n("in", "Input", {"shape": "1, 32", "dtype": "long", "name": "tokens"}, 0),
        _n("emb", "Embedding", {"num_embeddings": 1000, "embedding_dim": 64}, 1),
        _n("tb", "TransformerEncoderLayer", {"nhead": 8, "dim_feedforward": 256}, 2),
        _n("pool", "Mean", {"dim": 1}, 3),
        _n("cls", "Linear", {"out_features": 10}, 4),
        _n("out", "Output", {}, 5),
    ])
    dn, link = _dataset("model", y=0.0)
    return Project(
        models=[ModelDef(id="model", name="Model", graph=graph)],
        data_nodes=[dn], links=[link],
        training={"loss": "CrossEntropyLoss", "epochs": 10},
    )


def _gan() -> Project:
    gen = _chain([
        _n("in", "Input", {"shape": "1, 100"}, 0),
        _n("l1", "Linear", {"out_features": 256}, 1),
        _n("a1", "LeakyReLU", {"negative_slope": 0.2}, 2),
        _n("l2", "Linear", {"out_features": 784}, 3),
        _n("t", "Tanh", {}, 4),
        _n("out", "Output", {}, 5),
    ])
    disc = _chain([
        _n("in", "Input", {"shape": "1, 784"}, 0),
        _n("l1", "Linear", {"out_features": 256}, 1),
        _n("a1", "LeakyReLU", {"negative_slope": 0.2}, 2),
        _n("l2", "Linear", {"out_features": 1}, 3),
        _n("out", "Output", {}, 4),
    ])
    return Project(
        models=[
            ModelDef(id="g", name="Generator", graph=gen, sys_position=NodePosition(x=0, y=-90)),
            ModelDef(id="d", name="Discriminator", graph=disc, sys_position=NodePosition(x=280, y=90)),
        ],
        data_nodes=[
            DataNode(id="noise", kind="noise", name="Noise",
                     sys_position=NodePosition(x=-260, y=-90), config={"dims": "100", "distribution": "normal"}),
            DataNode(id="data", kind="dataset", name="Data",
                     sys_position=NodePosition(x=-260, y=90), config={"source": "memory"}),
        ],
        links=[
            ModelLink(id="noise-link", source_data="noise", target_model="g"),
            ModelLink(id="data-link", source_data="data", target_model="d"),
            ModelLink(id="gd-link", source_model="g", target_model="d"),
        ],
        training={
            "recipe": "gan", "epochs": 100,
            "roles": {"generator": "g", "discriminator": "d"},
            "per_role": {"generator": {"lr": 2e-4}, "discriminator": {"lr": 2e-4}},
        },
    )


def _vae() -> Project:
    # Encoder: a shared trunk forking into the two NAMED outputs the recipe
    # reads (mu / logvar).
    enc = Graph.model_validate({
        "nodes": [
            _n("in", "Input", {"shape": "1, 784"}, 0),
            _n("l1", "Linear", {"out_features": 400}, 1),
            _n("r1", "ReLU", {}, 2),
            _n("lmu", "Linear", {"out_features": 16}, 3, y=-80.0),
            _n("llv", "Linear", {"out_features": 16}, 3, y=80.0),
            _n("omu", "Output", {"name": "mu"}, 4, y=-80.0),
            _n("olv", "Output", {"name": "logvar"}, 4, y=80.0),
        ],
        "edges": [
            _e("in", "l1"), _e("l1", "r1"), _e("r1", "lmu"), _e("r1", "llv"),
            _e("lmu", "omu"), _e("llv", "olv"),
        ],
    })
    dec = _chain([
        _n("in", "Input", {"shape": "1, 16"}, 0),
        _n("l1", "Linear", {"out_features": 400}, 1),
        _n("r1", "ReLU", {}, 2),
        _n("l2", "Linear", {"out_features": 784}, 3),
        _n("s", "Sigmoid", {}, 4),
        _n("out", "Output", {}, 5),
    ])
    dn, link = _dataset("e", y=0.0)
    return Project(
        models=[
            ModelDef(id="e", name="Encoder", graph=enc, sys_position=NodePosition(x=0, y=-90)),
            ModelDef(id="d", name="Decoder", graph=dec, sys_position=NodePosition(x=0, y=90)),
        ],
        data_nodes=[dn], links=[link],
        training={"recipe": "vae", "epochs": 30, "roles": {"encoder": "e", "decoder": "d"}},
    )


@dataclass(frozen=True)
class TemplateDef:
    name: str
    label: str
    description: str
    build: Callable[[], Project]


TEMPLATES: dict[str, TemplateDef] = {
    t.name: t
    for t in (
        TemplateDef("mlp", "MLP classifier",
                    "The classic starter: 784 → 128 → 10 with a ReLU — MNIST-shaped.", _mlp),
        TemplateDef("cnn", "CNN classifier",
                    "Two Conv2d + MaxPool blocks into a linear head — images as images.", _cnn),
        TemplateDef("transformer", "Transformer classifier",
                    "Tokens → Embedding → Transformer Block → mean-pool → head.", _transformer),
        TemplateDef("gan", "GAN",
                    "Generator + Discriminator, noise and data pre-wired, adversarial recipe set.", _gan),
        TemplateDef("vae", "VAE",
                    "Encoder (named mu/logvar outputs) + Decoder, VAE recipe set.", _vae),
    )
}
