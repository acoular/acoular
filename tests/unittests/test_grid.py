# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for Grid classes and sector classes."""

from functools import partial
from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from pytest_cases import fixture, get_case_id, parametrize_with_cases

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


@fixture(scope='module')
def import_grid():
    """Fixture for creating a rectangular ImportGrid object with 4 grid points."""
    # Create 4 points with center at (0, 0, 0) and aperture of 2
    pos_total = np.array([[0, 1, 0, -1], [1, 0, -1, 0], [0, 0, 0, 0]])
    return ac.ImportGrid(pos=pos_total)


def test_load_xml():
    """Test if grid data is loaded correctly from an XML file."""
    xml_file_path = Path(ac.__file__).parent / 'xml' / 'array_56.xml'
    # Test for correct number of grid positions and shapes
    grid = ac.ImportGrid(file=xml_file_path)
    assert grid.size == 56
    assert grid.pos.shape == (3, 56)


@parametrize_with_cases('grid', cases=Grids)
def test_export_gpos_all_grids(grid, tmp_path):
    """Test export_gpos and import for all Grid-derived classes."""
    export_file = tmp_path / f'test_gpos_{grid.digest}.xml'
    grid.export_gpos(export_file)
    new_grid = ac.ImportGrid(file=export_file)
    assert np.array_equal(grid.pos, new_grid.pos)
