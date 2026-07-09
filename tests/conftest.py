from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Mirror the sys.path setup the scripts/ entrypoints perform, so the suite runs
# from a bare checkout as well as from an editable install. IsRankingRobust is
# needed by the parity test, which imports `package.RankAMIP`.
for path in (ROOT / "src", ROOT, ROOT / "IsRankingRobust"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: test downloads data from the Hugging Face Hub")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m"):
        return
    skip_network = pytest.mark.skip(reason="needs network; run with `-m network` to enable")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
