# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Cases for testing caching functionality of Acoular objects."""

import acoular as ac
import numpy as np
from pytest_cases import parametrize


class Caching:
    """Cases for testing caching functionality of Acoular objects.

    Current cases:
    - BeamformerBase, BeamformerSODIX, BeamformerGridlessOrth
    - PowerSpectra
    - PointSpreadFunction
    - Cache (sources: TimeSamples and RFFT)
    """

    @parametrize(
        'beamformer',
        [ac.BeamformerBase, ac.BeamformerSODIX, ac.BeamformerGridlessOrth],
        ids=['BeamformerBase', 'SODIX', 'GridlessOrth'],
    )
    @parametrize('cached', [False, True], ids=['cached-False', 'cached-True'])
    def case_caching_beamformer(self, small_source_case, beamformer, cached):
        bfs = []
        for _ in range(3):
            bf = beamformer(cached=cached, freq_data=small_source_case.freq_data, steer=small_source_case.steer)
            if hasattr(bf, 'n_iter'):
                bf.n_iter = 1
            bfs.append(bf)
            if hasattr(bf, 'shgo'):
                bf.shgo = {'n': 10, 'iters': 1}

        def calc(beamformer):
            return beamformer.synthetic(8000, 0)

        return bfs, calc, cached

    @parametrize('cached', [False, True], ids=['cached-False', 'cached-True'])
    @parametrize('trait', ['csm', 'eve', 'eva'])
    def case_caching_PowerSpectra(self, create_time_data_source, cached, trait):
        source = create_time_data_source(num_channels=2, num_samples=128)
        pss = []
        for _ in range(3):
            pss.append(ac.PowerSpectra(source=source, cached=cached, block_size=128))

        def calc(spectra):
            return getattr(spectra, trait)

        return pss, calc, cached

    @parametrize('calcmode', ['single', 'block', 'full', 'readonly'])
    def case_caching_PointSpreadFunction(self, small_source_case, calcmode):
        psfs = []
        for _ in range(3):
            psfs.append(
                ac.PointSpreadFunction(calcmode=calcmode, steer=small_source_case.steer, freq=8000, grid_indices=[0, 1])
            )

        def calc(point_spread_function):
            return point_spread_function.psf

        return psfs, calc, True

    def case_caching_time(self, create_time_data_source):
        source = create_time_data_source(num_channels=1, num_samples=2)
        tcs = []
        for _ in range(3):
            tcs.append(ac.Cache(source=source))

        def calc(cache):
            block_size = 1
            block_c = ac.tools.return_result(cache, num=block_size)
            block_nc = ac.tools.return_result(source, num=block_size)
            np.testing.assert_array_almost_equal(block_c, block_nc)
            # using unfinished cache
            block_c = ac.tools.return_result(cache, num=block_size)
            block_nc = ac.tools.return_result(source, num=block_size)
            np.testing.assert_array_almost_equal(block_c, block_nc)
            return block_c

        return tcs, calc, True

    def case_caching_spectra(self, create_time_data_source):
        source = create_time_data_source(num_channels=1, num_samples=4)
        source = ac.RFFT(source=source, block_size=2)
        tcs = []
        for _ in range(3):
            tcs.append(ac.Cache(source=source))

        def calc(cache):
            block_size = 1
            block_c = ac.tools.return_result(cache, num=block_size)
            block_nc = ac.tools.return_result(source, num=block_size)
            np.testing.assert_array_almost_equal(block_c, block_nc)
            # using unfinished cache
            block_c = ac.tools.return_result(cache, num=block_size)
            block_nc = ac.tools.return_result(source, num=block_size)
            np.testing.assert_array_almost_equal(block_c, block_nc)
            return block_c

        return tcs, calc, True
