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
        # Attention is permutation-invariant — without position information the
        # block is a bag-of-tokens mixer.
        _n("pos", "PositionalEmbedding", {"max_len": 128}, 2),
        _n("tb", "TransformerEncoderLayer", {"nhead": 8, "dim_feedforward": 256}, 3),
        _n("pool", "Mean", {"dim": 1}, 4),
        _n("cls", "Linear", {"out_features": 10}, 5),
        _n("out", "Output", {}, 6),
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


def _cond_graph(main_params: dict, tail: list[dict]) -> Graph:
    """A conditional model: a main Input and a ``label`` Input (embedded), joined
    by a Concat that then feeds ``tail`` (a positioned MLP ending in Output). The
    label rides its own port so the cGAN recipe can condition on the class."""
    head = [
        _n("main", "Input", main_params, 0, y=-40),
        _n("label", "Input", {"shape": "1", "dtype": "long", "name": "label"}, 0, y=80),
        _n("emb", "Embedding", {"num_embeddings": 10, "embedding_dim": 50}, 1, y=80),
        _n("cat", "Concat", {"dim": 1}, 2, y=0),
    ]
    edges = [
        _e("main", "cat", tgt_h="in0"),
        _e("label", "emb"),
        _e("emb", "cat", tgt_h="in1"),
        _e("cat", tail[0]["id"]),
        *[_e(a["id"], b["id"]) for a, b in zip(tail, tail[1:])],
    ]
    return Graph.model_validate({"nodes": head + tail, "edges": edges})


def _cgan() -> Project:
    gen = _cond_graph(
        {"shape": "1, 100", "name": "noise"},
        [
            _n("l1", "Linear", {"out_features": 256}, 3),
            _n("a1", "LeakyReLU", {"negative_slope": 0.2}, 4),
            _n("l2", "Linear", {"out_features": 512}, 5),
            _n("a2", "LeakyReLU", {"negative_slope": 0.2}, 6),
            _n("l3", "Linear", {"out_features": 784}, 7),
            _n("t", "Tanh", {}, 8),
            _n("out", "Output", {}, 9),
        ],
    )
    disc = _cond_graph(
        {"shape": "1, 784", "name": "image"},
        [
            _n("l1", "Linear", {"out_features": 512}, 3),
            _n("a1", "LeakyReLU", {"negative_slope": 0.2}, 4),
            _n("l2", "Linear", {"out_features": 256}, 5),
            _n("a2", "LeakyReLU", {"negative_slope": 0.2}, 6),
            _n("l3", "Linear", {"out_features": 1}, 7),
            _n("out", "Output", {}, 8),
        ],
    )
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
        # The label (y) conditions both models; X feeds the discriminator, noise the
        # generator — exactly what ensureCganWiring would provision.
        links=[
            ModelLink(id="noise-link", source_data="noise", target_model="g", target_input="main"),
            ModelLink(id="data-x-link", source_data="data", source_pin="x", target_model="d", target_input="main"),
            ModelLink(id="data-yg-link", source_data="data", source_pin="y", target_model="g", target_input="label"),
            ModelLink(id="data-yd-link", source_data="data", source_pin="y", target_model="d", target_input="label"),
        ],
        training={
            "recipe": "cgan", "epochs": 250,
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


def _reinforce() -> Project:
    # Policy net for CartPole: 4 observations → 2 action LOGITS (no softmax
    # head — the recipe samples Categorical(logits=…)). The env node is the
    # data source; the loop streams per-episode returns.
    graph = _chain([
        _n("in", "Input", {"shape": "1, 4", "name": "obs"}, 0),
        _n("l1", "Linear", {"out_features": 64}, 1),
        _n("r1", "ReLU", {}, 2),
        _n("l2", "Linear", {"out_features": 2}, 3),
        _n("out", "Output", {}, 4),
    ])
    env = DataNode(id="env", kind="env", name="CartPole",
                   sys_position=NodePosition(x=-260, y=0), config={"env_id": "CartPole-v1"})
    return Project(
        models=[ModelDef(id="model", name="Policy", graph=graph)],
        data_nodes=[env],
        links=[ModelLink(id="env-link", source_data="env", target_model="model")],
        # ~150 iterations shows real learning (the notebook-scale demo); the
        # curve visibly climbs within the first 30.
        training={"recipe": "reinforce", "epochs": 150, "roles": {"policy": "model"}},
    )


def _finetune() -> Project:
    # Transfer learning as the canvas sees it: the backbone yields FEATURES
    # (its classifier head is stripped) and the head you train is a node you
    # can see and resize. Frozen by default — the usual starting point, and
    # what makes this trainable on a laptop.
    graph = _chain([
        _n("in", "Input", {"shape": "1, 3, 224, 224", "name": "image"}, 0),
        _n("bb", "Backbone", {"arch": "resnet18", "pretrained": True, "freeze": True}, 1),
        _n("head", "Linear", {"out_features": 10}, 2),
        _n("out", "Output", {}, 3),
    ])
    # An image folder is where a fine-tuning set usually lives; the root is the
    # user's to fill in. ImageNet statistics because the weights were fitted to
    # them — the readiness panel says so if this is ever changed.
    dn = DataNode(id="data", kind="dataset", name="Photos",
                  sys_position=NodePosition(x=-260, y=0),
                  config={"source": "imagefolder", "root": "./data",
                          "resize": 224, "normalize": "imagenet", "val_split": 0.2})
    return Project(
        models=[ModelDef(id="model", name="Model", graph=graph)],
        data_nodes=[dn],
        links=[ModelLink(id="data-link", source_data="data", target_model="model")],
        training={"recipe": "supervised", "loss": "CrossEntropyLoss", "epochs": 5, "lr": 1e-3},
    )


def _language_model() -> Project:
    # A small GPT-shaped model: tokens → embedding → positions → ONE causal
    # block → logits over the vocabulary at every position. Causal is the
    # load-bearing setting — without it each position reads the token it's
    # being asked to predict.
    graph = _chain([
        _n("in", "Input", {"shape": "1, 64", "dtype": "long", "name": "tokens"}, 0),
        _n("emb", "Embedding", {"num_embeddings": 128, "embedding_dim": 128}, 1),
        _n("pos", "PositionalEmbedding", {"max_len": 256}, 2),
        _n("tb", "TransformerEncoderLayer",
           {"nhead": 4, "dim_feedforward": 256, "dropout": 0.1, "is_causal": True}, 3),
        _n("head", "Linear", {"out_features": 128}, 4),
        _n("out", "Output", {}, 5),
    ])
    # Unconfigured like every template's data node: register text with
    # sess.data(corpus=...) and pick it here. 128 is a placeholder vocabulary
    # (ASCII-ish) — readiness names the real size the moment text is picked.
    dn = DataNode(id="data", kind="dataset", name="Corpus",
                  sys_position=NodePosition(x=-260, y=0),
                  config={"source": "sequence", "block_size": 64, "val_split": 0.1})
    return Project(
        models=[ModelDef(id="model", name="LM", graph=graph)],
        data_nodes=[dn],
        links=[ModelLink(id="data-link", source_data="data", target_model="model")],
        training={"recipe": "causal_lm", "epochs": 20, "lr": 3e-4},
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
                    "Tokens → Embedding → positions → Transformer Block → mean-pool → head.", _transformer),
        TemplateDef("gan", "GAN",
                    "Generator + Discriminator, noise and data pre-wired, adversarial recipe set.", _gan),
        TemplateDef("cgan", "Conditional GAN",
                    "A GAN whose label conditions both models — Embedding + Concat, y fanned to both.", _cgan),
        TemplateDef("vae", "VAE",
                    "Encoder (named mu/logvar outputs) + Decoder, VAE recipe set.", _vae),
        TemplateDef("reinforce", "REINFORCE (CartPole)",
                    "A policy net balancing CartPole — RL as a recipe: env node wired, returns stream live.", _reinforce),
        TemplateDef("finetune", "Fine-tune a backbone",
                    "A frozen ImageNet resnet18 with your own head — point it at a folder of labelled images.",
                    _finetune),
        TemplateDef("languagemodel", "Language model (character)",
                    "Tokens → causal Transformer Block → next-token logits; register text and sample from it.",
                    _language_model),
    )
}
