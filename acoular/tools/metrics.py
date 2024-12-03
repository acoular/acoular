# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements metrics for evaluating signal processing results.

.. autosummary::
    :toctree: generated/

    MetricEvaluator
"""

from copy import copy

from numpy import (
    empty,
    inf,
    minimum,
    ones,
)
from scipy.spatial.distance import cdist
from traits.api import Bool, CArray, HasStrictTraits, Instance, Property

from acoular.fbeamform import L_p, integrate
from acoular.grids import CircSector, Grid


class MetricEvaluator(HasStrictTraits):
    """Evaluate the reconstruction performance of source mapping methods.

    This class can be used to calculate the following performance metrics
    according :cite:`Herold2017`:
    * Specific level error
    * Overall level error
    * Inverse level error
    """

    #: an array of shape=(nf,ng) containing the squared sound pressure data of the
    #: source mapping. (nf: number of frequencies, ng: number of grid points)
    data = CArray(shape=(None, None), desc='Contains the calculated squared sound pressure values in Pa**2.')

    #: an array of shape=(nf,ns) containing the squared sound pressure data of the
    #: ground-truth sources. (nf: number of frequencies, ns: number of sources)
    target_data = CArray(shape=(None, None), desc='Contains the ground-truth squared sound pressure values in Pa**2.')

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations
    #: for the calculated source mapping data.
    grid = Instance(Grid, desc='Grid instance that belongs to the calculated data')

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations
    #: for the ground-truth data.
    target_grid = Instance(Grid, desc='Grid instance that belongs to the ground-truth data')

    #: sector type. Currently only circular sectors are supported.
    sector = Instance(CircSector, default=CircSector())

    #: if set True: use shrink integration area if two sources are closer
    #: than 2*r. The radius of the integration area is then set to half the
    #: distance between the two sources.
    adaptive_sector_size = Bool(True, desc='adaptive integration area')

    #: if set `True`, the same amplitude can be assigned to multiple targets if
    #: the integration area overlaps. If set `False`, the amplitude is assigned
    #: to the first target and the other targets are ignored.
    multi_assignment = Bool(
        True,
        desc='if set True, the same amplitude can be assigned to multiple targets if the integration area overlaps',
    )

    #: returns the determined sector sizes for each ground-truth source position
    sectors = Property()

    def _validate_shapes(self):
        if self.data.shape[0] != self.target_data.shape[0]:
            msg = 'data and target_data must have the same number of frequencies!'
            raise ValueError(msg)
        if self.data.shape[1] != self.grid.size:
            msg = 'data and grid must have the same number of grid points!'
            raise ValueError(msg)
        if self.target_data.shape[1] != self.target_grid.size:
            msg = 'target_data and target_grid must have the same number of grid points!'
            raise ValueError(msg)

    def _get_sector_radii(self):
        ns = self.target_data.shape[1]
        radii = ones(ns) * self.sector.r
        if self.adaptive_sector_size:
            locs = self.target_grid.pos.T
            intersrcdist = cdist(locs, locs)
            intersrcdist[intersrcdist == 0] = inf
            intersrcdist = intersrcdist.min(0) / 2
            radii = minimum(radii, intersrcdist)
        return radii

    def _get_sectors(self):
        """Returns a list of CircSector objects for each target location."""
        r = self._get_sector_radii()
        ns = self.target_data.shape[1]
        sectors = []
        for i in range(ns):
            loc = self.target_grid.pos[:, i]
            sector = copy(self.sector)
            sector.r = r[i]
            sector.x = loc[0]
            sector.y = loc[1]
            sectors.append(sector)
        return sectors

    def _integrate_sectors(self):
        """Integrates over target sectors.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated Pa**2 values for each sector

        """
        sectors = self.sectors
        results = empty(shape=self.target_data.shape)
        for f in range(self.target_data.shape[0]):
            data = self.data[f]
            for i in range(self.target_data.shape[1]):
                sector = sectors[i]
                results[f, i] = integrate(data, self.grid, sector)
                if not self.multi_assignment:
                    indices = self.grid.subdomain(sector)
                    data[indices] = 0  # set values to zero (can not be assigned again)
        return results

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,)

        """
        self._validate_shapes()
        return L_p(self.data.sum(axis=1)) - L_p(self.target_data.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources

        """
        self._validate_shapes()
        sector_result = self._integrate_sectors()
        return L_p(sector_result) - L_p(self.target_data)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)

        """
        self._validate_shapes()
        sector_result = self._integrate_sectors()
        return L_p(sector_result.sum(axis=1)) - L_p(self.data.sum(axis=1))
