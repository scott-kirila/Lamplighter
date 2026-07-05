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
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    # Graph-global training config (loss/optimizer/hyperparams). Empty = defaults.
    # DEPRECATED: training/data are project-level concerns (see Project). The
    # single-model frontend still ships them on the graph, so they live here as a
    # compatibility shim, lifted to the Project by ``project_from_graph`` and
    # merged back by ``graph_from_project``. Phase B (project-native frontend)
    # removes them.
    training: dict[str, Any] = {}
    # Data-pipeline config (source, batching) driving the Data panel. Empty = defaults.
    data: dict[str, Any] = {}


class ModelDef(BaseModel):
    """One model in a project: a named graph with a spot on the system canvas.

    ``graph`` carries only nodes/edges — training and data are project-level (a
    project trains its models together under one recipe, on one data pipeline).
    ``name`` becomes the generated class name (sanitized), so distinct models
    read as ``class Generator(nn.Module)`` / ``class Discriminator(nn.Module)``.
    """

    id: str
    name: str
    graph: Graph = Graph()
    sys_position: NodePosition = NodePosition(x=0.0, y=0.0)


class ModelLink(BaseModel):
    """A dataflow claim on the system canvas, into a model's input port. The
    source is either another model's output (``source_model`` [+ ``source_pin``],
    e.g. a generator's samples into a discriminator) or a data node
    (``source_data``, e.g. MNIST or a noise source into a model). Exactly one of
    ``source_model`` / ``source_data`` is set. Shape-checked; the recipe reads
    links as evidence of how data and models compose."""

    id: str
    source_model: str | None = None  # a model id (model→model link)
    source_pin: str | None = None  # Output node id (None = the sole output)
    source_data: str | None = None  # a data-node id (data→model link)
    target_model: str
    target_input: str | None = None  # Input node id (None = the sole input)


class DataNode(BaseModel):
    """A data source on the system canvas, wired into a model's input port. A
    ``dataset`` node generates a ``make_dataloaders()`` / DataLoader — its
    ``config`` is the Data-panel form (source, batching, the picked variables);
    a ``noise`` node generates an in-loop sampler for a GAN's latent — its
    ``config`` carries the noise dims/distribution. Data becomes nodes you wire
    and configure in the Inspector rather than a single project-level Data tab."""

    id: str
    kind: str = "dataset"  # "dataset" | "noise"
    name: str = "Data"
    sys_position: NodePosition = NodePosition(x=0.0, y=0.0)
    # dataset: the DATA_PARAMS form dict; noise: {"dims": [...], "distribution": ...}.
    config: dict[str, Any] = {}


class Project(BaseModel):
    """The whole design: one or more models, the data sources feeding them, how
    everything connects, and the shared training config. A single-model project
    (the classic case) is just ``models=[one]`` with empty ``links`` — the
    adapters below make a lone Graph look like one, so single-model flows are
    unchanged."""

    version: int = 3
    models: list[ModelDef] = []
    # Data sources on the system canvas (dataset / noise), wired into model
    # inputs. Empty while the single project-level ``data`` form is still in use;
    # data nodes replace it as they roll out.
    data_nodes: list[DataNode] = []
    links: list[ModelLink] = []
    # {"recipe": "supervised", <recipe params>, "roles": {role: model_id},
    #  "per_role": {role: {...}}}. Empty = defaults + the supervised recipe.
    training: dict[str, Any] = {}


# The sole model's id in a single-model project (also the default ``moves``/shape
# key while the frontend is still single-model).
SOLE_MODEL_ID = "model"
# The materialized dataset node's id when adapting a bare single-model Graph.
SOLE_DATA_ID = "data"


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


def project_from_graph(graph: Graph) -> Project:
    """Wrap a single Graph as a one-model project: its nodes/edges become the sole
    model, its training rides the project, and its (single-model) data form becomes
    a dataset node wired into that model. The inverse of ``graph_from_project``.
    Every backend ingress that still receives a bare Graph runs this — bare
    notebook snippets, ``set_graph``, and single-model snapshot resume (whose Graph
    carries the resolved data config)."""
    inner = Graph(nodes=graph.nodes, edges=graph.edges)
    data_nodes: list[DataNode] = []
    links: list[ModelLink] = []
    if graph.data:
        data_nodes = [DataNode(id=SOLE_DATA_ID, kind="dataset", name="Data", config=dict(graph.data))]
        links = [ModelLink(id=f"{SOLE_DATA_ID}-link", source_data=SOLE_DATA_ID, target_model=SOLE_MODEL_ID)]
    return Project(
        models=[ModelDef(id=SOLE_MODEL_ID, name="Model", graph=inner)],
        training=dict(graph.training or {}),
        data_nodes=data_nodes,
        links=links,
    )


def graph_from_project(project: Project) -> Graph:
    """The single-model compatibility view: the (first) model's nodes/edges with
    the project's training merged back on and its wired data config as ``.data``,
    so every existing get_graph() caller sees exactly the Graph it always did.
    Returns an empty Graph for a project with no models."""
    if not project.models:
        return Graph(training=dict(project.training or {}))
    model = project.models[0]
    return Graph(
        nodes=model.graph.nodes,
        edges=model.graph.edges,
        training=dict(project.training or {}),
        data=resolve_data_config(project, model.id),
    )
