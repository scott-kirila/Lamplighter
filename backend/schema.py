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
    """A dataflow claim on the system canvas: one model's output feeds another's
    input (e.g. a generator's samples into a discriminator). Shape-checked; the
    recipe reads links as evidence of how the models compose."""

    id: str
    source_model: str
    source_pin: str | None = None  # Output node id (None = the sole output)
    target_model: str
    target_input: str | None = None  # Input node id (None = the sole input)


class Project(BaseModel):
    """The whole design: one or more models, how they connect, and the shared
    training/data config. A single-model project (the classic case) is just
    ``models=[one]`` with empty ``links`` — the adapters below make a lone Graph
    look like one, so single-model flows are unchanged."""

    version: int = 2
    models: list[ModelDef] = []
    links: list[ModelLink] = []
    # {"recipe": "supervised", <recipe params>, "roles": {role: model_id},
    #  "per_role": {role: {...}}}. Empty = defaults + the supervised recipe.
    training: dict[str, Any] = {}
    # Data-pipeline config (source, batching), as today. Empty = defaults.
    data: dict[str, Any] = {}


# The sole model's id in a single-model project (also the default ``moves``/shape
# key while the frontend is still single-model).
SOLE_MODEL_ID = "model"


def project_from_graph(graph: Graph) -> Project:
    """Wrap a single Graph as a one-model project, lifting its (deprecated)
    training/data onto the project. The inverse of ``graph_from_project`` for a
    one-model project. Every backend ingress point that still receives a bare
    Graph runs this, so old notebook snippets and stored snapshots keep working."""
    inner = Graph(nodes=graph.nodes, edges=graph.edges)
    return Project(
        models=[ModelDef(id=SOLE_MODEL_ID, name="Model", graph=inner)],
        training=dict(graph.training or {}),
        data=dict(graph.data or {}),
    )


def graph_from_project(project: Project) -> Graph:
    """The single-model compatibility view: the (first) model's nodes/edges with
    the project's training/data merged back on, so every existing get_graph()
    caller sees exactly the Graph it always did. Returns an empty Graph for a
    project with no models."""
    if not project.models:
        return Graph(training=dict(project.training or {}), data=dict(project.data or {}))
    model = project.models[0]
    return Graph(
        nodes=model.graph.nodes,
        edges=model.graph.edges,
        training=dict(project.training or {}),
        data=dict(project.data or {}),
    )
