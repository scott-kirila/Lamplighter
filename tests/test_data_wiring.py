"""The runner resolves a model's data by following the wires from data nodes —
the wired dataset node is the single source of truth. A dataset node wired into a
model feeds it; with nothing wired, the resolved config is empty."""
import torch

from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project, resolve_data_config
from lamplighter.backend.runner import RunManager
from tests.helpers import edge, graph, node


def _mlp():
    g = graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 3}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    return ModelDef(id="m", name="Model", graph=Graph(nodes=g.nodes, edges=g.edges))


def test_resolve_data_config_reads_the_wired_dataset_node():
    dn = DataNode(id="x", kind="dataset", name="Data", config={"source": "memory", "x_var": "X"})
    project = Project(
        models=[_mlp()],
        data_nodes=[dn],
        links=[ModelLink(id="L", source_data="x", target_model="m")],
    )
    assert resolve_data_config(project, "m") == {"source": "memory", "x_var": "X"}
    # No wire → no data config (the node is the only source).
    assert resolve_data_config(Project(models=[_mlp()]), "m") == {}


def test_supervised_run_follows_a_wired_dataset_node():
    dn = DataNode(
        id="x", kind="dataset", name="Data",
        config={"source": "memory", "x_var": "X", "y_var": "y", "batch_size": 4},
    )
    project = Project(
        models=[_mlp()],
        data_nodes=[dn],
        links=[ModelLink(id="L", source_data="x", target_model="m")],
        training={"recipe": "supervised", "epochs": 2, "device": "cpu", "seed": 0},
    )
    ns = {"X": torch.randn(20, 8), "y": torch.randint(0, 3, (20,))}
    mgr = RunManager()
    assert mgr.start(project, namespace=ns, emit=lambda m: None) is None
    assert mgr.join(timeout=30)
    assert mgr.state == "done", mgr.error
    assert len(mgr.history["train_loss"]) == 2

    # The snapshot recorded the wired node's config as the resolved data, and the
    # generated loader was built from it.
    assert mgr.snapshot["data"]["x_var"] == "X"
    assert mgr.snapshot["data"]["batch_size"] == 4
    assert "make_dataloaders" in mgr.snapshot["sources"]["data"]
