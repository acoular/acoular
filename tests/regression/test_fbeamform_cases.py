from copy import deepcopy

import acoular as ac
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

    @parametrize('steer_type', ['true level', 'classic', 'inverse', 'true location'], ids=[
        'true-level', 'classic', 'inverse', 'true-location'
    ])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerBase(self, source_case, steer_type, r_diag):
        steer = deepcopy(source_case.steer)
        steer.steer_type = steer_type
        return ac.BeamformerBase(
            freq_data=source_case.freq_data, steer=steer, r_diag=r_diag)

    @case(id="BfEig")
    def case_BeamformerEig(self, source_case):
        return ac.BeamformerEig(
            freq_data=source_case.freq_data,
            steer=source_case.steer,n=54,
            r_diag=False)

    @case(id="Music")
    def case_BeamformerMusic(self, source_case):
        return ac.BeamformerMusic(
            freq_data=source_case.freq_data,
            steer=source_case.steer, n=6)

    @case(id="Orth")
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerOrth(self, r_diag, source_case):
        return ac.BeamformerOrth(
            freq_data=source_case.freq_data, r_diag=r_diag,
            steer=source_case.steer, eva_list=list(range(38, 54)))

    @case(id="Damas")
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerDamas(self, r_diag, source_case):
        return ac.BeamformerDamas(
            freq_data=source_case.freq_data, r_diag=r_diag,
            steer=source_case.steer, n_iter=10, damp=0.98)

    @case(id="DamasPlus")
    @parametrize('method', ['NNLS', 'LP', 'LassoLars', 'OMPCV'])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerDamasPlus(self, method, r_diag, source_case):
        return ac.BeamformerDamasPlus(
            freq_data=source_case.freq_data, r_diag=r_diag,
            steer=source_case.steer, method=method,
            max_iter=10)

    @case(id="CleanSC")
    def case_BeamformerCleansc(self, source_case):
        return ac.BeamformerCleansc(
            freq_data=source_case.freq_data, r_diag=False,
            steer=source_case.steer, n=10, damp=0.9, stopn=4)

    @case(id="Clean")
    def case_BeamformerClean(self, source_case):
        return ac.BeamformerClean(
            freq_data=source_case.freq_data, r_diag=False,
            steer=source_case.steer, n_iter=10, damp=0.9)

    @case(id="Functional")
    def case_BeamformerFunctional(self, source_case):
        return ac.BeamformerFunctional(
            freq_data=source_case.freq_data, r_diag=False,
            steer=source_case.steer, gamma=3)

    @case(id="GIB")
    @parametrize('method', [
        'InverseIRLS', 'LassoLars', 'LassoLarsBIC', 'LassoLarsCV',
        'OMPCV', 'NNLS', 'Suzuki'
        ])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerGIB(self, method, r_diag, source_case):
        return ac.BeamformerGIB(n=10,
            r_diag=r_diag, freq_data=source_case.freq_data,
            steer=source_case.steer, method=method)

    @case(id="CMF")
    @parametrize('method', ['LassoLars', 'LassoLarsBIC', 'OMPCV', 'NNLS',
        'fmin_l_bfgs_b', 'Split_Bregman', 'FISTA'])
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerCMF(self, method, r_diag, source_case):
        return ac.BeamformerCMF(r_diag=r_diag,
            freq_data=source_case.freq_data, max_iter=10,
            steer=source_case.steer, method=method)

    @case(id="SODIX")
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerSODIX(self, r_diag, source_case):
        return ac.BeamformerSODIX(r_diag=r_diag,
            freq_data=source_case.freq_data,
            steer=source_case.steer, max_iter=10)

    @case(id="GridlessOrth")
    @parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
    def case_BeamformerGridlessOrth(self, r_diag, source_case):
        return ac.BeamformerGridlessOrth(r_diag=r_diag,
            freq_data=source_case.freq_data,
            steer=source_case.steer, n=1, shgo={'n': 16})

    @parametrize('bf', BF_DEFAULT)
    def case_default(self, bf, source_case):
        return bf(freq_data=source_case.freq_data, steer=source_case.steer)

