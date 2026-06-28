"""Tiny builders for constructing graphs in tests. Pydantic coerces the plain
dicts into the schema models, so tests read close to the on-the-wire JSON."""
from backend.schema import Graph


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
