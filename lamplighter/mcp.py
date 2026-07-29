"""MCP server: the pre-flight check as a tool coding agents call.

One tool, ``check_training``: the agent hands over a few lines of setup code
(build/import the model and data), the server runs them in a fresh
subprocess and returns :func:`lamplighter.check`'s report as JSON. The agent
that wrote the training code is the worst reviewer of it — this gives it a
verdict computed from the real tensors instead.

The subprocess never imports the lamplighter package: it loads
``backend/checks.py`` by file path (the module is deliberately
self-contained), so the interpreter being checked needs torch but not a
lamplighter install. That matters because the server may live in one
environment (wherever the MCP host installed it) and the user's project in
another — pass ``python=`` to point at the project's interpreter.

Run it with ``lamplighter-mcp`` or ``python -m lamplighter.mcp`` (stdio
transport; needs the ``lamplighter[mcp]`` extra). For Claude Code::

    claude mcp add lamplighter -- python -m lamplighter.mcp

Note the setup code EXECUTES with the caller's privileges — this is a local
development tool with the same trust level as the agent's own shell.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Separates the setup code's own stdout chatter from the result payload. The
# harness emits it exactly once, immediately before the JSON.
_SEP = chr(30)

# Runs inside the target interpreter. Loads the check engine by file path —
# no lamplighter import, no package machinery — then execs the setup source
# and checks whatever it named. Every outcome, including failure, leaves
# through emit(): one sentinel, one JSON document, exit 0.
_HARNESS = """\
import importlib.util, json, sys, traceback

SEP = chr(30)


def emit(payload):
    sys.stdout.write(SEP + json.dumps(payload))
    sys.stdout.flush()
    sys.exit(0)


checks_path, setup_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("_lamplighter_checks", checks_path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)

ns = {"__name__": "__lamplighter_setup__"}
try:
    with open(setup_path, encoding="utf-8") as f:
        source = f.read()
    exec(compile(source, "<setup>", "exec"), ns)
except (Exception, SystemExit):
    emit({"error": "the setup code raised before any check could run",
          "traceback": traceback.format_exc(limit=8)})

model, data = ns.get("model"), ns.get("data")
if model is None or data is None:
    missing = " and ".join(f"`{n}`" for n in ("model", "data") if ns.get(n) is None)
    emit({"error": f"the setup code must define {missing} — assign the nn.Module "
                   "to `model` and the DataLoader/Dataset/(X, y)/tensor to `data`"})

try:
    report = checks.check(model, data, ns.get("y"), loss=ns.get("loss"),
                          batch_size=ns.get("batch_size"))
except TypeError as exc:
    emit({"error": f"check() refused the setup's objects: {exc}"})
except (Exception, SystemExit):
    emit({"error": "check() itself failed — please report this",
          "traceback": traceback.format_exc(limit=8)})

emit(report.to_dict())
"""


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def run_setup_and_check(
    setup: str, *, cwd: str | None = None, python: str | None = None, timeout: int = 180
) -> dict[str, Any]:
    """Execute ``setup`` in a fresh interpreter and check what it built.

    Returns :meth:`CheckReport.to_dict`'s shape on success, or ``{"error":
    ...}`` (plus captured output) when the setup or the subprocess failed —
    always a dict, never an exception, so the tool result is uniform.
    """
    from .backend import checks as _checks_mod

    checks_path = str(Path(_checks_mod.__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="lamplighter-check-") as tmp:
        setup_path = str(Path(tmp) / "setup.py")
        Path(setup_path).write_text(setup, encoding="utf-8")
        try:
            proc = subprocess.run(
                [python or sys.executable, "-c", _HARNESS, checks_path, setup_path],
                capture_output=True, text=True, cwd=cwd, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {timeout}s — keep the setup to "
                             "construction (no training loops or large downloads), "
                             "or raise timeout="}
        except OSError as exc:
            return {"error": f"couldn't launch {python or sys.executable}: {exc}"}

    chatter, sep, payload = proc.stdout.rpartition(_SEP)
    if not sep:
        return {"error": f"the check subprocess exited with code {proc.returncode} "
                         "before producing a result",
                "stdout": _tail(proc.stdout), "stderr": _tail(proc.stderr)}
    try:
        result: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError:
        return {"error": "the check subprocess produced an unreadable result",
                "stdout": _tail(proc.stdout), "stderr": _tail(proc.stderr)}
    if "error" in result and proc.stderr:
        result["stderr"] = _tail(proc.stderr)
    if chatter.strip():
        result["setup_stdout"] = _tail(chatter)
    return result


_INSTRUCTIONS = """\
Lamplighter pre-flight-checks a PyTorch training setup against the REAL model
and data — one forward pass on a real batch, the actual label values, the
loader's actual arithmetic — and reports what will go wrong before any epoch
is spent. Call check_training BEFORE starting a training run, and again after
changing the model, the data pipeline, the loss, or the batch size. It reads
facts that reviewing the code cannot: several of the failures it catches raise
no error at all (out-of-range labels on Apple MPS silently produce a wrong
loss), and the rest crash only mid-run."""


def build_server():
    """The MCP server (import-time cheap; raises ImportError without the
    ``lamplighter[mcp]`` extra). Built on the 2.x SDK's ``MCPServer`` — the
    successor of 1.x's ``FastMCP``, same decorator surface."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("lamplighter", instructions=_INSTRUCTIONS)

    @server.tool()
    def check_training(
        setup: str, cwd: str | None = None, python: str | None = None, timeout: int = 180
    ) -> str:
        """Pre-flight a PyTorch training setup against the real model and data, before training.

        Runs `setup` (Python source) in a fresh subprocess, then checks the
        objects it built. The code must assign `model` (an nn.Module) and
        `data` (a DataLoader, Dataset, (X, y) tuple, tensor, or dict of
        tensors), and should also assign when known: `loss` (instance, class,
        or name), `y` (targets, when not inside `data`), `batch_size` (int,
        when `data` isn't already a DataLoader). Reuse the project's own
        pipeline by importing it:

            from train import build_model, make_loaders
            model = build_model()
            data, _ = make_loaders()
            import torch.nn as nn
            loss = nn.CrossEntropyLoss()

        Keep the setup to CONSTRUCTION — it executes for real, so no training
        loops or large downloads (there is a timeout).

        Catches silent failures a code review cannot: labels out of range of
        the output width (a mid-run CUDA assert — or a silently WRONG loss on
        Apple MPS), float labels under CrossEntropyLoss, (N, 1) column
        targets, a softmax stacked under CrossEntropyLoss (detected from the
        real outputs, so an F.softmax inside forward() counts), probabilities
        fed to NLLLoss, misaligned X/y counts, class imbalance, NaN/Inf
        outputs, reshapes that fold the batch dimension, and the
        final-batch-of-1 × BatchNorm crash.

        Returns JSON: {ok, errors, warnings, checks: [{level, title,
        detail}]} — level is "ok" | "warn" | "error", detail carries the fix.
        Fix every error before training; relay warnings to the user. `cwd`
        sets the working directory (project root) so imports resolve;
        `python` picks the interpreter when the project venv differs from the
        server's.
        """
        return json.dumps(
            run_setup_and_check(setup, cwd=cwd, python=python, timeout=timeout),
            indent=2, ensure_ascii=False,
        )

    return server


def main() -> None:
    """Console entry point (``lamplighter-mcp``): serve on stdio."""
    try:
        server = build_server()
    except ImportError as exc:
        import importlib.util

        # Two different problems, two different fixes — telling someone who
        # HAS the SDK to install the extra sends them in a circle.
        if importlib.util.find_spec("mcp") is None:
            sys.exit('the MCP server needs the optional extra: pip install "lamplighter[mcp]"')
        sys.exit(f"the installed mcp SDK is incompatible with this server ({exc}) — "
                 'lamplighter needs the 2.x SDK: pip install "mcp>=2,<3"')
    server.run()


if __name__ == "__main__":
    main()
