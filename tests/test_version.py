"""lamplighter.__version__ — the runtime version marker a bug report starts
with. In this checkout (dev installs are deps-only, the package is never
installed) it resolves via the pyproject fallback; installed wheels resolve via
importlib.metadata. Either way it must match pyproject exactly."""
import tomllib
from pathlib import Path

import lamplighter


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert lamplighter.__version__ == declared
    assert "__version__" in lamplighter.__all__
