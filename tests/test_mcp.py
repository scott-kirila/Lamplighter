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


def test_server_exposes_the_tool():
    pytest.importorskip("mcp")
    import asyncio

    from lamplighter.mcp import build_server

    tools = asyncio.run(build_server().list_tools())
    (tool,) = tools
    assert tool.name == "check_training"
    for needle in ("model", "data", "loss", "MPS", "before training"):
        assert needle in tool.description
