# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implementation of test cases for all grids and sectors."""

import acoular as ac
import numpy as np
from pytest_cases import parametrize

from tests.utils import get_subclasses

SECTOR_SKIP_DEFAULT = [
    ac.Sector,
    ac.CircSector,
    ac.RectSector,
    ac.RectSector3D,
    ac.PolySector,
    ac.ConvexSector,
    ac.MultiSector,
]

GRIDS_SKIP_DEFAULT = [
    ac.Grid,
    ac.RectGrid,
    ac.RectGrid3D,
    ac.LineGrid,
    ac.ImportGrid,
    ac.MergeGrid,
    ac.BeamformerAdaptiveGrid,
    ac.BeamformerGridlessOrth,
]

SECTOR_DEFAULT = [s for s in get_subclasses(ac.Sector) if s not in SECTOR_SKIP_DEFAULT]
GRIDS_DEFAULT = [g for g in get_subclasses(ac.Grid) if g not in GRIDS_SKIP_DEFAULT]


class Grids:
    """Test cases for all grids.

    New grids should be added here. If no dedicated test case is added for a :class:`Grid` derived
    class, the `case_default` case will raise a `NotImplementedError`. If a dedicated test case was
    added for a grid, it should be added to the `GRIDS_SKIP_DEFAULT` list, which excludes the class
    from `case_default`.

    New cases should create an instance of the grid with at least one grid-point at (0, 0, 1) and
    return it. The grid should not exceed the bounds of (-1, -1, 1) and (1, 1, 1) in x, y, and z
    direction, respectively.
    """

    def case_RectGrid(self):
        return ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1, increment=1)

    def case_RectGrid3D(self):
        return ac.RectGrid3D(x_min=-1, x_max=1, y_min=-1, y_max=1, z_min=1, z_max=1, increment=1)

    def case_LineGrid(self):
        return ac.LineGrid(loc=(-1, 0, 1), length=2, num_points=3)

    def case_ImportGrid(self):
        return ac.ImportGrid(pos=ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1, increment=1).pos)

    def case_MergeGrid(self):
        return ac.MergeGrid(
            grids=[
                ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1, increment=1),
                ac.LineGrid(loc=(-1, 0, 1), length=2, num_points=3),
            ]
        )


if len(GRIDS_DEFAULT) > 0:

    @parametrize('grid', GRIDS_DEFAULT)
    def case_default(self, grid):  # noqa: ARG001
        msg = f'Please write a test case for class {grid.__name__}'
        raise NotImplementedError(msg)

    Grids.case_default = case_default


class Sectors:
    """Test cases for all sectors.

    New sectors should be added here. If no dedicated test case is added for a :class:`Sector`
    derived class, the `case_default` case will raise a `NotImplementedError`. If a dedicated test
    case was added for a sector, it should be added to the `SECTOR_SKIP_DEFAULT` list, which
    excludes the class from `case_default`.

    For a new sector, two cases should be added: one for a sector containing the point (0, 0, 1) and
    one for an empty sector, not laying in the region of (-1, -1, 1) and (1, 1, 1) in x, y, and z
    direction, respectively.

    """

    def case_numpy_array2D(self):
        return np.array([0, 0, 0.2])

    def case_numpy_array3D(self):
        return np.array([0, 0, 1, 0, 0, 1])

    def case_RectSector_full(self):
        return ac.RectSector(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2)

    def case_RectSector_empty(self):
        off = 10  # some offset to make the sector empty
        return ac.RectSector(
            x_min=-0.2 + off,
            x_max=0.2 + off,
            y_min=-0.2 + off,
            y_max=0.2 + off,
            include_border=False,
            default_nearest=False,
        )

    def case_CircSector_full(self):
        return ac.CircSector(x=0, y=0, r=0.2)

    def case_CircSector_empty(self):
        off = 10  # some offset to make the sector empty
        return ac.CircSector(x=off, y=off, r=0.2, include_border=False, default_nearest=False)

    def case_RectSector3D_full(self):
        return ac.RectSector3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=1, z_max=1)

    def case_RectSector3D_empty(self):
        off = 10  # some offset to make the sector empty
        return ac.RectSector3D(
            x_min=-0.2 + off,
            x_max=0.2 + off,
            y_min=-0.2 + off,
            y_max=0.2 + off,
            z_min=1 + off,
            z_max=1 + off,
            include_border=False,
            default_nearest=False,
        )

    def case_PolySector_full(self):
        return ac.PolySector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2])

    def case_PolySector_empty(self):
        off = 10  # some offset to make the sector empty
        return ac.PolySector(
            edges=[0.2 + off, 0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off],
            include_border=False,
            default_nearest=False,
        )

    def case_ConvexSector_full(self):
        return ac.ConvexSector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2])

    def case_ConvexSector_empty(self):
        off = 10  # some offset to make the sector empty
        return ac.ConvexSector(
            edges=[0.2 + off, 0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off, -0.2 + off, 0.2 + off, -0.2 + off],
            include_border=False,
            default_nearest=False,
        )

    def case_MultiSector_full(self):
        sectors = [
            ac.CircSector(x=0, y=0, r=0.2),
            ac.RectSector(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2),
            ac.RectSector3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=1, z_max=1),
            ac.PolySector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2]),
            ac.ConvexSector(edges=[0.2, 0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.2]),
        ]
        return ac.MultiSector(sectors=sectors)

    def case_MultiSector_empty(self):
        off = 10  # some offset to make the sector empty
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
        return ac.MultiSector(sectors=sectors)


if len(SECTOR_DEFAULT) > 0:

    @parametrize('sector', SECTOR_DEFAULT)
    def case_default(self, sector):  # noqa: ARG001
        msg = f'Please write a test case for class {sector.__name__}'
        raise NotImplementedError(msg)

    Sectors.case_default = case_default
