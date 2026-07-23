"""The pydantic domain models — the serialization shapes of a project.

``Graph`` (nodes/edges), ``ModelDef``, ``DataNode``, ``ModelLink``, and ``Project``
are the wire/persistence format, mirrored field-for-field by the frontend's
``Domain*`` types so load/save is a straight JSON pass-through. ``Project`` is the
source of truth end to end; ``resolve_data_config`` reads the data feeding a model
off its wired dataset node.
"""
from pydantic import BaseModel
from typing import Any


class NodePosition(BaseModel):
    x: float
    y: float


class GraphNode(BaseModel):
    id: str
    type: str
    position: NodePosition
    params: dict[str, Any] = {}


class GraphEdge(BaseModel):
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class Graph(BaseModel):
    """One model's structure: nodes + edges. Training and data are project-level
    concerns (see Project) — a Graph carries neither."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class ImportInfo(BaseModel):
    """Provenance for a model brought in via ``sess.inspect(model)``. It carries
    only the ordered source ``state_dict`` keys — enough to seed the generated
    module positionally and assert shape-by-shape at run start — NOT the weight
    tensors themselves, which stay in the kernel (a resnet50 state_dict is 100MB,
    and the autosaved graph.json is neither the place nor fsync'd). The runner
    resolves the live weights by model id from the datastore's import registry."""

    source: str                      # the original class name, for display
    state_keys: list[str] = []       # ordered original state_dict keys


class ModelDef(BaseModel):
    """One model in a project: a named graph with a spot on the overview canvas.

    ``graph`` carries only nodes/edges — training and data are project-level (a
    project trains its models together under one recipe, on one data pipeline).
    ``name`` becomes the generated class name (sanitized), so distinct models
    read as ``class Generator(nn.Module)`` / ``class Discriminator(nn.Module)``.
    ``imported`` is set when the graph came from ``sess.inspect`` — the runner
    seeds it with the original weights before the first run.
    """

    id: str
    name: str
    graph: Graph = Graph()
    sys_position: NodePosition = NodePosition(x=0.0, y=0.0)
    imported: ImportInfo | None = None


class ModelLink(BaseModel):
    """A dataflow claim on the overview canvas, into a model's input port. The
    source is either another model's output (``source_model`` [+ ``source_pin``],
    e.g. a generator's samples into a discriminator) or a data node
    (``source_data``, e.g. MNIST or a noise source into a model). Exactly one of
    ``source_model`` / ``source_data`` is set. Shape-checked; the recipe reads
    links as evidence of how data and models compose."""

    id: str
    source_model: str | None = None  # a model id (model→model link)
    # For a model source: the Output node id (None = the sole output). For a data
    # source: the dataset's output pin — "x" (features, the default) or "y" (a
    # labeled dataset's targets, e.g. a cGAN's class label). None = "x".
    source_pin: str | None = None
    source_data: str | None = None  # a data-node id (data→model link)
    target_model: str
    target_input: str | None = None  # Input node id (None = the sole input)


class DataNode(BaseModel):
    """A data source on the overview canvas, wired into a model's input port. A
    ``dataset`` node generates a ``make_dataloaders()`` / DataLoader — its
    ``config`` is the Data-panel form (source, batching, the picked variables);
    a ``noise`` node generates an in-loop sampler for a GAN's latent — its
    ``config`` carries the noise dims/distribution. Data becomes nodes you wire
    and configure in the Inspector rather than a single project-level Data tab."""

    id: str
    kind: str = "dataset"  # "dataset" | "noise" | "env"
    name: str = "Data"
    sys_position: NodePosition = NodePosition(x=0.0, y=0.0)
    # dataset: the DATA_PARAMS form dict; noise: {"dims": [...], "distribution": ...}.
    config: dict[str, Any] = {}


class Project(BaseModel):
    """The whole design: one or more models, the data sources feeding them, how
    everything connects, and the shared training config. A single-model project
    (the classic case) is just ``models=[one]`` with empty ``links``."""

    version: int = 3
    models: list[ModelDef] = []
    # Data sources on the overview canvas (dataset / noise), wired into model
    # inputs via ``links``.
    data_nodes: list[DataNode] = []
    links: list[ModelLink] = []
    # {"recipe": "supervised", <recipe params>, "roles": {role: model_id},
    #  "per_role": {role: {...}}}. Empty = defaults + the supervised recipe.
    training: dict[str, Any] = {}


# The sole model's id in a single-model project (also the default ``moves``/shape
# key while the frontend is still single-model).
SOLE_MODEL_ID = "model"


def resolve_data_config(project: Project, model_id: str | None) -> dict[str, Any]:
    """The data config feeding a model: the ``config`` of the dataset node wired
    into it, or ``{}`` when nothing is wired. Data lives on the wired node — the
    single source of truth."""
    for link in project.links:
        if link.source_data is not None and link.target_model == model_id:
            dn = next(
                (d for d in project.data_nodes if d.id == link.source_data and d.kind == "dataset"), None
            )
            if dn is not None:
                return dict(dn.config or {})
    return {}


def resolve_env_config(project: Project, model_id: str | None) -> dict[str, Any] | None:
    """The environment feeding a policy (an RL recipe's data source): the
    ``config`` of the ``env``-kind node wired into the model, or None when
    nothing is wired — the dataset resolver's sibling, same single-source rule."""
    for link in project.links:
        if link.source_data is not None and link.target_model == model_id:
            dn = next(
                (d for d in project.data_nodes if d.id == link.source_data and d.kind == "env"), None
            )
            if dn is not None:
                return dict(dn.config or {})
    return None
