import acoular as ac
import numpy as np
import pytest
from pytest_cases import fixture, get_case_id, parametrize, parametrize_with_cases
from test_grid_cases import Grids, Sectors


@fixture(scope='session')
def setup_mics_integrate():
    rng1 = np.random.RandomState(1)
    return ac.MicGeom(mpos_tot=rng1.normal(size=3 * 5).reshape((3, 5)))


@fixture(scope='session')
@parametrize('f', [2000], ids=['2kHz'])
def setup_power_spectra_integrate(setup_mics_integrate, f):
    mics = setup_mics_integrate
    steer = ac.SteeringVector(grid=ac.ImportGrid(gpos_file=np.array([[0], [0], [1]])), mics=mics)
    H = np.empty((1, mics.num_mics, 1), dtype=complex)
    H[0] = steer.transfer(f).T
    csm = H @ H.swapaxes(2, 1).conjugate()
    return ac.PowerSpectraImport(csm=csm, frequencies=f)


@fixture(scope='session')
@parametrize_with_cases('grid', cases=Grids)
def setup_beamformer_integrate(setup_power_spectra_integrate, setup_mics_integrate, grid):
    steer = ac.SteeringVector(grid=grid, mics=setup_mics_integrate)
    bf = ac.BeamformerBase(freq_data=setup_power_spectra_integrate, steer=steer)
    return bf, grid


def skip_or_fail(grid, sector):
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
    bf, grid = setup_beamformer_integrate
    skip_or_fail(grid, sector)
    f = bf.freq_data.frequencies
    bf_res = bf.synthetic(f)
    integration_res = bf.integrate(sector)
    assert integration_res.shape == (1,)
    assert integration_res[0] == bf_res.max()
