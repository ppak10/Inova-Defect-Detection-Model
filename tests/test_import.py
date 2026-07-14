import importlib

import pytest

MODULES = [
    "runs.v03.constants",
    "runs.v03.dataset",
    "runs.v03.model",
    "runs.v03.trainer",
    "runs.v03.train",
    "runs.v03.evaluate",
    "runs.v03.visualize",
    "runs.v02.constants",
    "runs.v02.dataset",
    "runs.v02.model",
    "runs.v02.trainer",
    "runs.v02.train",
    "runs.v02.evaluate",
    "runs.v02.visualize",
    "runs.v01.constants",
    "runs.v01.prepare",
    "runs.v01.dataset",
    "runs.v01.registration",
    "runs.v01.model",
    "runs.v01.trainer",
    "runs.v01.train",
    "runs.v01.evaluate",
    "runs.v01.visualize",
    "runs.v01.export",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module: str) -> None:
    importlib.import_module(module)
