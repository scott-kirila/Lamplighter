"""Tiny builders for constructing graphs in tests. Pydantic coerces the plain
dicts into the schema models, so tests read close to the on-the-wire JSON."""
from backend.schema import DataNode, Graph, ModelDef, ModelLink, Project, SOLE_MODEL_ID


def single_model_project(g, training=None, data=None, model_id=SOLE_MODEL_ID):
    """A one-model Project from a Graph, with optional project-level ``training``
    and a wired dataset node carrying ``data`` — the test-side equivalent of the
    retired ``project_from_graph``, so single-model tests read naturally now that
    training/data are project concerns, not graph fields."""
    data_nodes, links = [], []
    if data:
        data_nodes = [DataNode(id="data", kind="dataset", name="Data", config=dict(data))]
        links = [ModelLink(id="data-link", source_data="data", target_model=model_id)]
    return Project(
        models=[ModelDef(id=model_id, name="Model", graph=g)],
        training=dict(training or {}),
        data_nodes=data_nodes,
        links=links,
    )


def node(nid, ntype, params=None, x=0.0, y=0.0):
    return {"id": nid, "type": ntype, "position": {"x": x, "y": y}, "params": params or {}}


def edge(src, tgt, src_h="output", tgt_h="input", eid=None):
    return {
        "id": eid or f"{src}->{tgt}:{tgt_h}",
        "source": src,
        "sourceHandle": src_h,
        "target": tgt,
        "targetHandle": tgt_h,
    }


def graph(nodes, edges):
    return Graph(nodes=nodes, edges=edges)


def output_id(g):
    return next(n.id for n in g.nodes if n.type == "Output")
