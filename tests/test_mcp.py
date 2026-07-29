"""The MCP surface: setup code in, check report out — through a real
subprocess, the way an agent's call travels. The subprocess loads the check
engine by file path (never `import lamplighter`), so these tests also prove
``backend/checks.py`` stays self-contained: the moment it grows a relative
import, every test here breaks.
"""
import pytest

from lamplighter.mcp import run_setup_and_check

_PLANTED_BUG = """\
import torch
from torch import nn

torch.manual_seed(0)  # the subprocess has fresh RNG — unseeded labels would flake
model = nn.Sequential(nn.Linear(8, 3), nn.Softmax(dim=-1))
data = (torch.randn(30, 8), torch.randint(1, 4, (30,)))  # labels 1…3, head of 3
loss = nn.CrossEntropyLoss()
"""


def test_planted_bugs_travel_end_to_end():
    result = run_setup_and_check(_PLANTED_BUG)
    assert result["ok"] is False
    titles = " | ".join(c["title"] for c in result["checks"] if c["level"] == "error")
    assert "has classes 1…3 but the model outputs 3" in titles
    assert "outputs probabilities but CrossEntropyLoss" in titles


def test_setup_stdout_is_kept_apart_from_the_result():
    result = run_setup_and_check("print('loading...')\n" + _PLANTED_BUG)
    assert result["ok"] is False  # chatter didn't corrupt the payload
    assert "loading..." in result.get("setup_stdout", "")


def test_missing_names_get_the_contract():
    result = run_setup_and_check("x = 1")
    assert "must define `model` and `data`" in result["error"]


def test_setup_crash_is_reported_with_traceback():
    result = run_setup_and_check("1 / 0")
    assert "raised before any check could run" in result["error"]
    assert "ZeroDivisionError" in result["traceback"]


def test_bad_interpreter_is_an_error_dict():
    result = run_setup_and_check("model = 1", python="/nonexistent/python")
    assert "couldn't launch" in result["error"]


def test_engine_works_where_lamplighter_is_not_importable(tmp_path):
    """The shipped promise: the checked interpreter needs torch, NOT a
    lamplighter install. With cwd pointed away from the repo, `import
    lamplighter` cannot resolve in the subprocess (the package is never
    pip-installed in dev or CI) — so this fails the moment checks.py grows
    ANY lamplighter import, absolute or relative. The plain tests only catch
    the relative kind, because they inherit the repo as cwd."""
    result = run_setup_and_check(_PLANTED_BUG, cwd=str(tmp_path))
    assert result["ok"] is False
    titles = " | ".join(c["title"] for c in result["checks"])
    assert "has classes 1…3 but the model outputs 3" in titles


def test_non_module_model_is_refused_with_the_reason():
    result = run_setup_and_check("import torch\nmodel = 1\ndata = torch.randn(4, 2)")
    assert "check() refused" in result["error"]
    assert "nn.Module" in result["error"]


def test_timeout_is_an_error_dict():
    result = run_setup_and_check("import time; time.sleep(30)", timeout=2)
    assert "timed out after 2s" in result["error"]


def test_silent_subprocess_is_an_error_dict():
    # An interpreter that launches fine but never emits the sentinel.
    result = run_setup_and_check("model = 1", python="/usr/bin/true")
    assert "before producing a result" in result["error"]


def test_server_exposes_the_tool():
    pytest.importorskip("mcp")
    import asyncio

    from lamplighter.mcp import build_server

    tools = asyncio.run(build_server().list_tools())
    (tool,) = tools
    assert tool.name == "check_training"
    # Pin the parts of the description that carry the contract, not just words
    # any docstring would contain.
    for needle in ("must assign `model`", "`data`", "loss", "MPS", "before training"):
        assert needle in tool.description


def test_tool_call_travels_through_the_server():
    """The closure itself — argument forwarding and JSON serialization — not
    just the listing. A broken tool body passes list_tools() fine."""
    pytest.importorskip("mcp")
    import asyncio
    import json

    from lamplighter.mcp import build_server

    result = asyncio.run(build_server().call_tool("check_training", {"setup": _PLANTED_BUG}))
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is False
    titles = " | ".join(c["title"] for c in payload["checks"])
    assert "has classes 1…3 but the model outputs 3" in titles
