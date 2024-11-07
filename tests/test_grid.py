from copy import deepcopy

import acoular as ac
import numpy as np
import pytest


def get_sector_classes():
    # for later testing condition: sector only includes (0,0,1) point
    sectors = [
        ac.CircSector(x=0, y=0, r=0.2),
        ac.RectSector(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2),
        ac.RectSector3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=1, z_max=1),
        ac.PolySector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2]),
        ac.ConvexSector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2]),
    ]
    multi_sector = ac.MultiSector(sectors=deepcopy(sectors))
    return sectors + [multi_sector]


def get_emtpy_sector_classes():
    # for later testing condition: sector should not cover any grid point
    off = 10
    sectors = [
        ac.CircSector(x=off, y=off, r=0.2, include_border=False, default_nearest=False),
        ac.RectSector(
            x_min=-0.2 + off,
            x_max=0.2 + off,
            y_min=-0.2 + off,
            y_max=0.2 + off,
            include_border=False,
            default_nearest=False,
        ),
        ac.RectSector3D(
            x_min=-0.2 + off,
            x_max=0.2 + off,
            y_min=-0.2 + off,
            y_max=0.2 + off,
            z_min=1 + off,
            z_max=1 + off,
            include_border=False,
            default_nearest=False,
        ),
        ac.PolySector(
            edges=[0.2 + off, 0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off],
            include_border=False,
            default_nearest=False,
        ),
        ac.ConvexSector(
            edges=[0.2 + off, 0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off],
            include_border=False,
            default_nearest=False,
        ),
    ]
    multi_sector = ac.MultiSector(sectors=deepcopy(sectors))
    return sectors + [multi_sector]


def get_rectgrid():
    return ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1, increment=1)


def get_rectgrid3D():
    return ac.RectGrid3D(x_min=-1, x_max=1, y_min=-1, y_max=1, z_min=1, z_max=1, increment=1)


def get_linegrid():
    return ac.LineGrid(loc=(-1, 0, 1), length=2, numpoints=3)


def get_importgrid():
    return ac.ImportGrid(gpos_file=get_rectgrid().gpos)


def get_mergegrid():
    return ac.MergeGrid(grids=[get_rectgrid(), get_linegrid()])


all_grids = [
    get_rectgrid(),
    get_rectgrid3D(),
    get_linegrid(),
    get_importgrid(),
    get_mergegrid(),
]

all_grids_and_sectors = [(grid, sector) for grid in all_grids for sector in get_sector_classes()]

all_grids_and_empty_sectors = [(grid, sector) for grid in all_grids for sector in get_emtpy_sector_classes()]


@pytest.mark.parametrize('grid', all_grids)
def test_size(grid):
    np.testing.assert_equal(grid.size, grid.gpos.shape[-1])


@pytest.mark.parametrize('grid, sector', all_grids_and_sectors)
def test_existing_subdomain(grid, sector):
    indices = grid.subdomain(sector)
    np.testing.assert_equal(indices[0].shape[0], 1)


@pytest.mark.parametrize('grid, sector', all_grids_and_empty_sectors)
def test_empty_subdomain(grid, sector):
    indices = grid.subdomain(sector)
    np.testing.assert_equal(indices[0].shape[0], 0)
    if hasattr(sector, 'default_nearest'):
        sector.default_nearest = True
        indices = grid.subdomain(sector)
        np.testing.assert_equal(indices[0].shape[0], 1)
