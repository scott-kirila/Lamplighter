"""Project schema — the wire/persistence shape.

``Project`` is the source of truth end to end (training and data are project
concerns; a ``Graph`` is just nodes + edges). These pin the model defaults, v3
shape, and that a pre-data-node ``v2`` dict still loads."""
from lamplighter.backend.codegen import generate_module
from lamplighter.backend.schema import DataNode, Graph, Project
from tests.helpers import edge, graph, node


def test_data_node_defaults_and_round_trip():
    d = DataNode(id="d0", kind="dataset", name="MNIST", config={"source": "torchvision"})
    assert d.model_dump() == {
        "id": "d0", "kind": "dataset", "name": "MNIST",
        "sys_position": {"x": 0.0, "y": 0.0}, "config": {"source": "torchvision"},
    }
    # A lone-arg DataNode (id only) is a dataset by default.
    assert DataNode(id="x").kind == "dataset"
    noise = DataNode(id="n", kind="noise", name="Noise", config={"dims": [100]})
    assert noise.kind == "noise" and noise.config["dims"] == [100]


def test_project_carries_data_nodes_and_is_v3():
    project = Project(models=[], data_nodes=[DataNode(id="d0", name="Data")])
    assert project.version == 3
    assert [n.id for n in project.data_nodes] == ["d0"]


def test_graph_has_no_training_or_data_fields():
    # The shim is gone — a Graph is nodes + edges only, and a stored graph with
    # leftover training/data keys still loads (the extra keys are ignored).
    g = Graph.model_validate({"nodes": [], "edges": [], "training": {"lr": 0.1}, "data": {}})
    assert not hasattr(g, "training")
    assert not hasattr(g, "data")


def test_v2_project_dict_still_validates():
    # A project stored before data nodes existed (no data_nodes key, version 2)
    # must load cleanly — data_nodes defaults to empty, nothing else changes.
    v2 = {"version": 2, "models": [], "links": [], "training": {"lr": 0.1}, "data": {"source": "memory"}}
    project = Project.model_validate(v2)
    assert project.data_nodes == []
    assert project.version == 2  # the stored value is preserved (informational)
    assert project.training == {"lr": 0.1}
    assert not hasattr(project, "data")  # the deprecated project-level form is gone (ignored on load)


def _mlp():
    return graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )


def test_class_name_default_is_byte_identical():
    g = _mlp()
    assert generate_module(g) == generate_module(g, class_name="GeneratedModel")


def test_class_name_renames_only_the_class():
    g = _mlp()
    default = generate_module(g)
    named = generate_module(g, class_name="Generator")
    assert "class Generator(nn.Module):" in named
    assert "class GeneratedModel(nn.Module):" not in named
    # Only the class line differs — the body is otherwise the same source.
    assert named.replace("class Generator(nn.Module):", "class GeneratedModel(nn.Module):") == default
