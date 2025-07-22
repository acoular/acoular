# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests integration of beamformer results over sectors."""

import acoular as ac
import numpy as np
import pytest
from pytest_cases import fixture, get_case_id, parametrize, parametrize_with_cases

from tests.cases.test_grid_cases import Grids, Sectors


@fixture(scope='session')
def setup_mics_integrate():
    """Fixture for setting up a microphone geometry with 5 microphones."""
    rng1 = np.random.RandomState(1)
    return ac.MicGeom(pos_total=rng1.normal(size=3 * 5).reshape((3, 5)))


@fixture(scope='session')
@parametrize('f', [2000], ids=['2kHz'])
def setup_power_spectra_integrate(setup_mics_integrate, f):
    """
    Fixture for setting up a power spectra object with 5 microphones and CSM for a single frequency.

    Parameters
    ----------
    setup_mics_integrate : instance of acoular.mics.MicGeom
        Microphone geometry (fixture)
    f : int
        Frequency in Hz
    """
    mics = setup_mics_integrate
    steer = ac.SteeringVector(grid=ac.ImportGrid(pos=np.array([[0], [0], [1]])), mics=mics)
    transf = np.empty((1, mics.num_mics, 1), dtype=complex)
    transf[0] = steer.transfer(f).T
    csm = transf @ transf.swapaxes(2, 1).conjugate()
    return ac.PowerSpectraImport(csm=csm, frequencies=f)


@fixture(scope='function')
@parametrize_with_cases('grid', cases=Grids)
def setup_beamformer_integrate(setup_power_spectra_integrate, setup_mics_integrate, grid):
    """
    Fixture for setting up a beamformer with a grid and power spectra object.

    Parameters
    ----------
    setup_power_spectra_integrate : instance of acoular.spectra.PowerSpectra
        Power spectra object (fixture)
    setup_mics_integrate : acoular.mics.MicGeom
        Microphone geometry (fixture)
    grid : acoular.grids.Grid
        Grid object (fixture)
    """
    steer = ac.SteeringVector(grid=grid, mics=setup_mics_integrate)
    bf = ac.BeamformerBase(freq_data=setup_power_spectra_integrate, steer=steer)
    return bf, grid


def skip_or_fail(grid, sector):
    """Define conditions to skip or fail tests based on grid and sector types."""
    if grid.__class__.__name__ in ['LineGrid', 'MergeGrid', 'ImportGrid'] and not isinstance(sector, ac.Sector):
        pytest.xfail(f'Grid {grid.__class__.__name__} has no indices method')

    if isinstance(grid, ac.RectGrid3D) and isinstance(sector, np.ndarray) and sector.shape[0] == 3:
        pytest.skip('RectGrid3D does not support 2D sectors')

    if isinstance(grid, ac.RectGrid) and isinstance(sector, np.ndarray) and sector.shape[0] > 4:
        pytest.skip('RectGrid does not support 3D sectors')


@parametrize_with_cases(
    'sector', cases=Sectors, filter=lambda cf: 'full' in get_case_id(cf) or 'numpy' in get_case_id(cf)
)
def test_sector_integration_functional(setup_beamformer_integrate, sector):
    """
    Test the integration of a beamformer result over a sector using integrate function.

    Parameters
    ----------
    setup_beamformer_integrate : tuple
        Beamformer and grid objects (fixture)
    sector : numpy.ndarray
        Sector indices (non-empty cases from Sectors)
    """
    bf, grid = setup_beamformer_integrate
    skip_or_fail(grid, sector)
    f = bf.freq_data.frequencies
    bf_res = bf.synthetic(f)
    integration_res = ac.integrate(data=bf_res, sector=sector, grid=grid)
    assert integration_res.shape == ()
    assert integration_res == bf_res.max()


@parametrize_with_cases(
    'sector', cases=Sectors, filter=lambda cf: 'full' in get_case_id(cf) or 'numpy' in get_case_id(cf)
)
def test_sector_integration(setup_beamformer_integrate, sector):
    """
    Test the integration of a beamformer result over a sector using integrate method.

    Parameters
    ----------
    setup_beamformer_integrate : tuple
        Beamformer and grid objects (fixture)
    sector : numpy.ndarray
        Sector indices (non-empty cases from Sectors)
    """
    bf, grid = setup_beamformer_integrate
    skip_or_fail(grid, sector)
    f = bf.freq_data.frequencies
    integration_res = bf.integrate(sector)
    bf_res = bf.synthetic(f)
    assert integration_res.shape == (1,)
    assert integration_res[0] == bf_res.max()
