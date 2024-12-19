# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for Grid classes and sector classes."""

from functools import partial

import pytest
from pytest_cases import get_case_id, parametrize_with_cases

from tests.cases.test_grid_cases import Grids, Sectors
from tests.utils import sector_case_filter


@parametrize_with_cases('grid', cases=Grids)
def test_grid_size(grid):
    """
    Test if the size of the grid matches the number of grid points.

    Parameters
    ----------
    grid : acoular.grids.Grid
        Grid instance to be tested
    """
    assert grid.size == grid.pos.shape[-1], 'Size of grid does not match number of grid points'


@parametrize_with_cases('grid', cases=Grids)
@parametrize_with_cases('sector', cases=Sectors, filter=partial(sector_case_filter, t='full'))
def test_subdomain(grid, sector):
    """Test if the sector is not empty as expected.

    Parameters
    ----------
    grid : acoular.grids.Grid
        Grid instance to be tested
    sector : acoular.grids.Sector
        Sector instance to be tested
    """
    assert grid.subdomain(sector)[0].shape[0] == 1, 'Subdomain is empty'


@parametrize_with_cases('grid', cases=Grids)
@parametrize_with_cases('sector', cases=Sectors, filter=lambda cf: 'empty' in get_case_id(cf))
def test_default_nearest(grid, sector):
    """Verify that the nearest grid point is assigned to the sector if default_nearest is True.

    Parameters
    ----------
    grid : acoular.grids.Grid
        Grid instance to be tested
    sector : acoular.grids.Sector
        Sector instance to be tested
    """
    if hasattr(sector, 'default_nearest'):
        sector.default_nearest = True
        assert grid.subdomain(sector)[0].shape[0] == 1, 'Subdomain is empty although default_nearest is set to True'
    else:
        pytest.skip('Sector does not have default_nearest attribute')


@parametrize_with_cases('grid', cases=Grids)
@parametrize_with_cases('sector', cases=Sectors, filter=partial(sector_case_filter, t='empty'))
def test_empty_subdomain(grid, sector):
    """Test if the sector is empty as expected.

    Parameters
    ----------
    grid : acoular.grids.Grid
        Grid instance to be tested
    sector : acoular.grids.Sector
        Sector instance to be tested
    """
    assert grid.subdomain(sector)[0].shape[0] == 0, 'Subdomain is not empty'
