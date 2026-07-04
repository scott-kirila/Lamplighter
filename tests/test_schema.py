"""Project schema + the single-model compatibility adapters.

A lone Graph must look like a one-model Project and round-trip back unchanged,
so every existing single-model flow (get_graph/set_graph, codegen, the runner)
is untouched while the Project becomes the backend's source of truth."""
from backend.codegen import generate_module
from backend.schema import (
    SOLE_MODEL_ID,
    Graph,
    Project,
    graph_from_project,
    project_from_graph,
)
from tests.helpers import edge, graph, node


def _mlp():
    return graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )


def test_project_from_graph_lifts_training_and_data():
    g = _mlp()
    g.training = {"epochs": 5, "lr": 0.1}
    g.data = {"source": "memory", "val_split": 0.2}

    project = project_from_graph(g)
    assert [m.id for m in project.models] == [SOLE_MODEL_ID]
    assert project.links == []
    # training/data live on the project now, not the inner model graph.
    assert project.training == {"epochs": 5, "lr": 0.1}
    assert project.data == {"source": "memory", "val_split": 0.2}
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
