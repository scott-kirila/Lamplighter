"""The generated-code trust boundary, in executable form.

Lamplighter runs the code it generates (one audited chokepoint:
``codegen.exec_generated``). The security boundary is therefore not exec itself
but what can flow *into* a template: interpolated values must be repr()-escaped,
identifiers validated, and enum-ish names checked against the registry. These
tests push hostile strings through every user-controllable string param and
assert they stay inert — the answer, as a regression suite, to "isn't exec
dangerous?"."""
import ast
import traceback

import pytest

from backend.codegen import (
    exec_generated,
    generate_dataloader,
    generate_module,
    generate_training,
    sanitize_class_name,
)
from backend.inference import graph_issues
from backend.schema import Graph
from tests.helpers import edge, graph, node, single_model_project

EVIL = "'); import os; os.system('boom') #"


def _mlp():
    return graph(
        [node("in", "Input", {"shape": "1, 8"}), node("l", "Linear", {"out_features": 4}),
         node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )


# --- imports allowlist: nothing generated may import outside torch/torchvision --

ALLOWED_IMPORT_ROOTS = {"torch", "torchvision"}


def _assert_imports_allowlisted(source: str):
    """Parse generated source and assert every import stays inside the expected
    libraries — the tripwire that catches a template-injection regression (an
    injected `import os` shows up here as a hard failure)."""
    tree = ast.parse(source)
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            roots = {alias.name.split(".")[0] for alias in stmt.names}
        elif isinstance(stmt, ast.ImportFrom):
            roots = {(stmt.module or "").split(".")[0]}
        else:
            continue
        assert roots <= ALLOWED_IMPORT_ROOTS, f"unexpected import in generated source: {roots}"


def test_generated_sources_import_only_torch_libraries():
    from backend.recipes import RECIPES

    g = _mlp()
    data = {"source": "memory", "val_split": 0.2}
    _assert_imports_allowlisted(generate_module(g))
    _assert_imports_allowlisted(generate_training(g, {}))
    _assert_imports_allowlisted(generate_dataloader(g, data))
    # Every recipe's trainer too (gan/cgan generate from a project).
    for recipe in RECIPES.values():
        _assert_imports_allowlisted(recipe.generate(single_model_project(g, data=data)))


# --- hostile params stay inert ------------------------------------------------

def test_torchvision_dataset_name_is_validated_not_interpolated():
    # The dataset name lands as an attribute (datasets.MNIST) — the one spot
    # repr() can't guard — so codegen re-checks the registry enum.
    with pytest.raises(ValueError, match="unknown torchvision dataset"):
        generate_dataloader(Graph(), {"source": "torchvision", "dataset": f"MNIST{EVIL}"})
    # The legitimate names still pass.
    src = generate_dataloader(Graph(), {"source": "torchvision", "dataset": "CIFAR10"})
    assert "datasets.CIFAR10(" in src


def test_hostile_root_path_stays_a_string_literal():
    src = generate_dataloader(Graph(), {"source": "imagefolder", "root": EVIL, "val_split": 0.2})
    compile(src, "<gen>", "exec")  # syntactically intact — nothing broke out
    assert repr(EVIL) in src  # the payload sits inside an escaped literal
    _assert_imports_allowlisted(src)  # ...and injected no import


def test_hostile_input_name_is_refused_before_any_exec():
    # A malicious Input name could smuggle code into forward()'s signature —
    # it's rejected as a graph issue, which the runner checks before codegen.
    g = graph(
        [node("in", "Input", {"shape": "1, 8", "name": "x, evil=__import__('os')"}),
         node("l", "Linear", {"out_features": 4}), node("out", "Output")],
        [edge("in", "l"), edge("l", "out")],
    )
    issues = graph_issues(g)
    assert any("not a valid identifier" in i for i in issues)

    from backend.runner import RunManager

    err = RunManager().start(single_model_project(g), namespace={}, emit=lambda m: None)
    assert err is not None and "identifier" in err  # refused, never exec'd


def test_hostile_model_name_sanitizes_to_a_plain_identifier():
    cls = sanitize_class_name(f"My Model {EVIL}")
    assert cls.isidentifier()
    src = generate_module(_mlp(), class_name=cls)
    compile(src, "<gen>", "exec")
    _assert_imports_allowlisted(src)


# --- the chokepoint itself ------------------------------------------------------

def test_exec_generated_tracebacks_show_the_generated_line():
    # linecache registration: an error raised inside generated code must show
    # the offending source line, not an opaque <lamplighter-...> stub.
    ns = exec_generated("def f():\n    raise ValueError('boom')\n", "<lamplighter-test-tb>")
    try:
        ns["f"]()
    except ValueError:
        tb = traceback.format_exc()
    assert '<lamplighter-test-tb>' in tb
    assert "raise ValueError('boom')" in tb  # the actual line, via linecache
