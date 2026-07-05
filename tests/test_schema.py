"""Project schema + the single-model compatibility adapters.

A lone Graph must look like a one-model Project and round-trip back unchanged,
so every existing single-model flow (get_graph/set_graph, codegen, the runner)
is untouched while the Project becomes the backend's source of truth."""
from backend.codegen import generate_module
from backend.schema import (
    SOLE_MODEL_ID,
    DataNode,
    Graph,
    Project,
    graph_from_project,
    project_from_graph,
)
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


def test_v2_project_dict_still_validates():
    # A project stored before data nodes existed (no data_nodes key, version 2)
    # must load cleanly — data_nodes defaults to empty, nothing else changes.
    v2 = {"version": 2, "models": [], "links": [], "training": {"lr": 0.1}, "data": {"source": "memory"}}
    project = Project.model_validate(v2)
    assert project.data_nodes == []
    assert project.version == 2  # the stored value is preserved (informational)
    assert project.training == {"lr": 0.1}
    assert not hasattr(project, "data")  # the deprecated project-level form is gone (ignored on load)


def test_project_from_graph_has_no_data_nodes():
    project = project_from_graph(_mlp())
    assert project.data_nodes == []
    assert project.version == 3


def _mlp():
    return graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )


def test_project_from_graph_lifts_training_and_materializes_a_dataset_node():
    g = _mlp()
    g.training = {"epochs": 5, "lr": 0.1}
    g.data = {"source": "memory", "val_split": 0.2}

    project = project_from_graph(g)
    assert [m.id for m in project.models] == [SOLE_MODEL_ID]
    # training rides the project; the data form becomes a dataset node wired in.
    assert project.training == {"epochs": 5, "lr": 0.1}
    assert len(project.data_nodes) == 1
    dn = project.data_nodes[0]
    assert dn.kind == "dataset" and dn.config == {"source": "memory", "val_split": 0.2}
    assert [(lk.source_data, lk.target_model) for lk in project.links] == [(dn.id, SOLE_MODEL_ID)]
    assert project.models[0].graph.training == {}
    assert project.models[0].graph.data == {}


def test_graph_round_trips_through_a_project():
    g = _mlp()
    g.training = {"epochs": 7}
    g.data = {"source": "memory"}
    assert graph_from_project(project_from_graph(g)).model_dump() == g.model_dump()


def test_graph_from_empty_project_is_empty():
    assert graph_from_project(Project()).model_dump() == Graph().model_dump()


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
