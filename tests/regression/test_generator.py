# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Snapshot tests for all generators."""

from pytest_cases import parametrize_with_cases

from tests.cases.test_generator_cases import Generators
from tests.utils import get_result


@parametrize_with_cases('obj', cases=Generators)
def test_generators(snapshot, obj):
    """Performs snapshot testing with snapshot fixture from pytest-regtest.

    Uses the generator cases defined in class Generators from test_generator_cases.py

    To overwrite the collected snapshots, run:

    ```bash
    pytest -v --regtest-reset tests/regression/test_generator.py::test_generators
    ```

    Parameters
    ----------
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    obj : instance of acoular.base.Generator
        Generator instance to be tested (cases from Generators)
    """
    result = get_result(obj, num=1)
    snapshot.check(result, rtol=5e-5, atol=5e-8)
