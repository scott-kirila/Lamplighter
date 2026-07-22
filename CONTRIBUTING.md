# Contributing

## Getting set up

```bash
uv sync --dev                    # Python deps (does NOT install the package —
                                 # the dev flow runs from the checkout)
cd frontend && npm install && npm run build
```

The backend serves `frontend/dist/`, so after editing frontend source run
`npm run build` and hard-refresh. Backend edits need a kernel restart — the
kernel caches imported modules.

```bash
uv run pytest -q                 # backend
uv run ruff check lamplighter/ tests/
cd frontend && npm test && npm run build   # build doubles as the typecheck
```

Notebooks under `examples/` are committed without outputs. Run
`uv run nbstripout --install --attributes .gitattributes` once per clone.

## The one architectural rule

**Nodes and recipes are declarative data; the engines are generic over them.**

A new layer is one `NodeDef` in `lamplighter/backend/registry.py`. A new
training loop is one `RecipeDef` in `recipes.py`. Shape inference (`inference.py`)
and code generation (`codegen.py`) consume that data — they do not branch on
node type or recipe name. 37 of 43 node types are pure data today, and
`tests/test_nodes.py` pins the partition so a per-type branch in an engine
cannot creep in unnoticed.

If you find yourself writing `if node.type == "..."` inside an engine, that is
the signal the registry is missing a field, not that the engine needs a case.

## What a change is expected to carry

This codebase leans on test-enforced invariants rather than review vigilance, so
the bar is less "add a test" than "make the thing that broke unable to break
silently again":

- **A bug fix pins the bug.** Not the fix's implementation — the behaviour.
- **A generated-code change goes through the oracle.** `tests/test_codegen_runtime.py`
  generates source, `exec`s it, runs a real tensor through, and asserts the
  output shape matches what `infer_shapes` predicted. That pattern is why
  codegen and inference cannot silently disagree.
- **A palette or recipe addition rides the existing parametrized suites.**
  `tests/test_templates.py` and `test_nodes.py` sweep every registry entry, so
  new entries are covered by adding data, not tests.
- **No `time.sleep` in tests.** Runs are driven from the emit callback — see
  the note at the top of `tests/test_runner.py`. The suite is ~640 tests in
  ~10s with no flakes, and that is worth protecting.

## Comments

The comment layer here records *decisions*, not mechanics — why `always_emit`
exists, why `convnext_tiny` is deliberately absent from the backbone list, why
the test split is carved last. That is what makes 1600-line files navigable.
Please write that kind of comment, and please don't write the other kind.

## Before opening a PR

Run the four commands above. If you touched anything colour-related, note that
`frontend/src/lib/contrast.test.ts` computes contrast ratios from `index.css`
itself — comments claiming a ratio are not trusted.

Please open an issue before a large change. This is a solo project with a
specific scope (a **local, single-user prototyping tool** — not a training
platform, not multi-tenant, not for distributed or LLM-scale models), and it is
kinder to disagree about direction before you've written the code.
