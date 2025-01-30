# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implement test cases for frequency-domain beamformers."""

from copy import deepcopy

import acoular as ac
import pytest
from pytest_cases import case, parametrize

from tests.utils import get_subclasses

# skip beamformers that have a dedicated test case in the
# Beamformer class and require additional parameters to work properly
# or are computationally demanding so that using the default does not make sense
BF_SKIP_DEFAULT = [
    ac.BeamformerBase,
    ac.BeamformerMusic,
    ac.BeamformerOrth,
    ac.BeamformerGIB,
    ac.BeamformerCMF,
    ac.BeamformerDamas,
    ac.BeamformerDamasPlus,
    ac.BeamformerSODIX,
    ac.BeamformerGridlessOrth,
    ac.BeamformerAdaptiveGrid,
]

BF_DEFAULT = [bf for bf in get_subclasses(ac.BeamformerBase) if bf not in BF_SKIP_DEFAULT]


class Beamformer:
    """Test cases for all beamformers.

    New beamformers should be added here. If no dedicated test case is added for a
    :class:`acoular.fbeamform.BeamformerBase` derived class, the class is still included in the test
    suite through the use of the `case_default` case. If a dedicated test case was added for a
    beamformer, it should be added to the `BF_SKIP_DEFAULT` list, which excludes the class from
    `case_default`.

    """

    @parametrize(
        'steer_type',
        ['true level', 'classic', 'inverse', 'true location'],
        ids=['true-level', 'classic', 'inverse', 'true-location'],
    )
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerBase(self, regression_source_case, steer_type, r_diag):
        steer = deepcopy(regression_source_case.steer)
        steer.steer_type = steer_type
        return ac.BeamformerBase(freq_data=regression_source_case.freq_data_import, steer=steer, r_diag=r_diag)

    @case(id='BfEig')
    def case_BeamformerEig(self, regression_source_case):
        return ac.BeamformerEig(
            freq_data=regression_source_case.freq_data_import, steer=regression_source_case.steer, n=54, r_diag=False
        )

    @case(id='Music')
    def case_BeamformerMusic(self, regression_source_case):
        return ac.BeamformerMusic(
            freq_data=regression_source_case.freq_data_import, steer=regression_source_case.steer, n=6
        )

    @case(id='Orth')
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerOrth(self, r_diag, regression_source_case):
        return ac.BeamformerOrth(
            freq_data=regression_source_case.freq_data_import,
            r_diag=r_diag,
            steer=regression_source_case.steer,
            eva_list=list(range(38, 54)),
        )

    @case(id='Damas')
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerDamas(self, r_diag, regression_source_case):
        return ac.BeamformerDamas(
            freq_data=regression_source_case.freq_data_import,
            r_diag=r_diag,
            steer=regression_source_case.steer,
            n_iter=10,
            damp=0.98,
        )

    @case(id='DamasPlus')
    @parametrize('method', ['NNLS', 'LP', 'LassoLars', 'OMPCV'])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerDamasPlus(self, method, r_diag, regression_source_case):
        if method == 'LP' and r_diag:
            pytest.skip('fails for LP and r_diag=True')
        return ac.BeamformerDamasPlus(
            freq_data=regression_source_case.freq_data_import,
            r_diag=r_diag,
            steer=regression_source_case.steer,
            method=method,
            n_iter=10,
        )

    @case(id='CleanSC')
    def case_BeamformerCleansc(self, regression_source_case):
        return ac.BeamformerCleansc(
            freq_data=regression_source_case.freq_data_import,
            r_diag=False,
            steer=regression_source_case.steer,
            n_iter=10,
            damp=0.9,
            stopn=4,
        )

    @case(id='Clean')
    def case_BeamformerClean(self, regression_source_case):
        return ac.BeamformerClean(
            freq_data=regression_source_case.freq_data_import,
            r_diag=False,
            steer=regression_source_case.steer,
            n_iter=10,
            damp=0.9,
        )

    @case(id='Functional')
    def case_BeamformerFunctional(self, regression_source_case):
        return ac.BeamformerFunctional(
            freq_data=regression_source_case.freq_data_import, r_diag=False, steer=regression_source_case.steer, gamma=3
        )

    @case(id='GIB')
    @parametrize('method', ['InverseIRLS', 'LassoLars', 'LassoLarsBIC', 'LassoLarsCV', 'OMPCV', 'NNLS', 'Suzuki'])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerGIB(self, method, r_diag, regression_source_case):
        return ac.BeamformerGIB(
            n=10,
            r_diag=r_diag,
            freq_data=regression_source_case.freq_data_import,
            steer=regression_source_case.steer,
            method=method,
            eps_perc=0.001,
            n_iter=5,
        )

    @case(id='CMF')
    @parametrize('method', ['LassoLars', 'LassoLarsBIC', 'OMPCV', 'NNLS', 'fmin_l_bfgs_b', 'Split_Bregman', 'FISTA'])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerCMF(self, method, r_diag, regression_source_case):
        return ac.BeamformerCMF(
            r_diag=r_diag,
            freq_data=regression_source_case.freq_data_import,
            n_iter=10,
            steer=regression_source_case.steer,
            method=method,
        )

    @case(id='SODIX')
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerSODIX(self, r_diag, regression_source_case):
        return ac.BeamformerSODIX(
            r_diag=r_diag,
            freq_data=regression_source_case.freq_data_import,
            steer=regression_source_case.steer,
            n_iter=10,
        )

    @case(id='GridlessOrth')
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerGridlessOrth(self, r_diag, regression_source_case):
        return ac.BeamformerGridlessOrth(
            r_diag=r_diag,
            freq_data=regression_source_case.freq_data_import,
            steer=regression_source_case.steer,
            n=1,
            shgo={'n': 16},
        )

    @parametrize('bf', BF_DEFAULT)
    def case_default(self, bf, regression_source_case):
        return bf(freq_data=regression_source_case.freq_data_import, steer=regression_source_case.steer)
