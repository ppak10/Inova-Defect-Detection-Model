import importlib

import pytest

MODULES = [
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
