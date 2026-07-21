"""The built-in templates — every one must be a WORKING project.

This is the drift-guard: a registry change (a renamed param, a new required
field) that breaks a template fails here instead of greeting a user with a
broken canvas. Each template must infer shapes without a single node error,
generate code for every model, and satisfy its recipe's expectations.
"""
import pytest

from lamplighter.backend.codegen import class_name_for, generate_module
from lamplighter.backend.inference import graph_issues, infer_shapes
from lamplighter.backend.recipes import get_recipe
from lamplighter.backend.templates import TEMPLATES


@pytest.mark.parametrize("name", list(TEMPLATES), ids=list(TEMPLATES))
def test_template_is_a_working_project(name):
    t = TEMPLATES[name]
    assert t.label and t.description
    project = t.build()
    assert project.models, f"{name}: no models"

    sole = len(project.models) <= 1
    for m in project.models:
        # Structurally valid…
        assert graph_issues(m.graph) == [], f"{name}/{m.name}: {graph_issues(m.graph)}"
        # …shapes infer with zero node errors…
        _, errors = infer_shapes(m.graph)
        assert errors == {}, f"{name}/{m.name}: {errors}"
        # …and the model generates compilable code.
        src = generate_module(m.graph, class_name=class_name_for(m.name, sole))
        compile(src, f"<{name}/{m.name}>", "exec")

    # The recipe accepts the project as-configured (roles included) — the
    # trainer generates without touching a namespace.
    recipe = get_recipe((project.training or {}).get("recipe"))
    assert recipe is not None
    recipe.generate(project)


def test_template_endpoints():
    from fastapi.testclient import TestClient

    from lamplighter.backend.app import app

    with TestClient(app) as c:
        listing = c.get("/api/templates").json()["templates"]
        assert [t["name"] for t in listing] == list(TEMPLATES)
        assert all(t["label"] and t["description"] for t in listing)

        gan = c.get("/api/templates/gan").json()
        assert [m["name"] for m in gan["models"]] == ["Generator", "Discriminator"]
        assert gan["training"]["recipe"] == "gan"

        assert c.get("/api/templates/nope").status_code == 404


def test_gan_template_link_evidence_is_clean():
    # The pre-wired generator→discriminator link must shape-check (784 = 784) —
    # a template should never open with a red wire.
    from lamplighter.backend.inference import link_issues, primary_shapes

    project = TEMPLATES["gan"].build()
    shapes = {}
    for m in project.models:
        s, _ = infer_shapes(m.graph)
        shapes[m.id] = primary_shapes(m.graph, s)
    results = {r["id"]: r for r in link_issues(project, shapes)}
    assert results["gd-link"]["ok"] is True
    assert "N × 784" in results["gd-link"]["message"]


def test_finetune_template_is_a_working_transfer_setup():
    # The template exists to make transfer learning discoverable, so the parts
    # that make it WORK (not merely build) are pinned.
    from lamplighter.backend.diagnose import diagnose

    project = TEMPLATES["finetune"].build()
    src = generate_module(project.models[0].graph)
    assert "models.resnet18(weights=ResNet18_Weights.DEFAULT)" in src
    assert "p.requires_grad = False" in src  # frozen: trains the head only
    # The head is sized from the backbone's probed feature width, not a guess.
    assert "nn.Linear(512, 10)" in src

    data = project.data_nodes[0].config
    assert data["normalize"] == "imagenet"  # the statistics those weights want
    assert data["resize"] == 224
    # Nothing is wrong with it out of the box — only the folder is the user's.
    assert [c for c in diagnose(project, {}) if c["level"] == "error"] == []


def test_language_model_template_masks_the_future():
    from lamplighter.backend.diagnose import diagnose

    project = TEMPLATES["languagemodel"].build()
    graph = project.models[0].graph
    block = next(n for n in graph.nodes if n.type == "TransformerEncoderLayer")
    assert block.params["is_causal"] is True  # the load-bearing setting

    src = generate_module(graph)
    assert "generate_square_subsequent_mask" in src
    # Logits over the whole vocabulary at every position, and a window that
    # matches what the loader yields.
    emb = next(n for n in graph.nodes if n.type == "Embedding")
    head = next(n for n in graph.nodes if n.type == "Linear")
    assert head.params["out_features"] == emb.params["num_embeddings"]
    inp = next(n for n in graph.nodes if n.type == "Input")
    assert inp.params["shape"] == f"1, {project.data_nodes[0].config['block_size']}"

    # The only thing missing is the user's text.
    errors = [c["title"] for c in diagnose(project, {}) if c["level"] == "error"]
    assert errors == ["Corpus: nothing picked"]
