import acoular as ac
import pytest
from pytest_cases import parametrize
from utils import get_beamformer_classes


class Beamformer:

    def case_BeamformerEig(self, source_case):
        return ac.BeamformerEig(
            freq_data=source_case.freq_data,
            steer=source_case.steer,n=54,
            )

    def case_BeamformerMusic(self, source_case):
        return ac.BeamformerMusic(
            freq_data=source_case.freq_data,
            steer=source_case.steer, n=6)

    def case_BeamformerOrth(self, source_case):
        return ac.BeamformerOrth(
            freq_data=source_case.freq_data,
            steer=source_case.steer, eva_list=list(range(38, 54)))

    def case_BeamformerDamas(self, source_case):
        return ac.BeamformerDamas(
            freq_data=source_case.freq_data,
            steer=source_case.steer, n_iter=10, damp=0.98)

    @parametrize('method', ['NNLS', 'LP', 'LassoLars', 'OMPCV'])
    def case_BeamformerDamasPlus(self, method, source_case):
        return ac.BeamformerDamasPlus(
            freq_data=source_case.freq_data,
            steer=source_case.steer, method=method,
            max_iter=10, alpha=10**-12, unit_mult=1e10)

    def case_BeamformerCleansc(self, source_case):
        return ac.BeamformerCleansc(
            freq_data=source_case.freq_data,
            steer=source_case.steer, n=10, damp=0.9, stopn=4)

    def case_BeamformerClean(self, source_case):
        return ac.BeamformerClean(
            freq_data=source_case.freq_data,
            steer=source_case.steer, n_iter=10, damp=0.9)

    def case_BeamformerFunctional(self, source_case):
        return ac.BeamformerFunctional(
            freq_data=source_case.freq_data,
            steer=source_case.steer, gamma=3)

    @parametrize('method', [
        'InverseIRLS', 'LassoLars', 'LassoLarsBIC', 'LassoLarsCV',
        'OMPCV', 'NNLS', 'Suzuki'
        ])
    def case_BeamformerGIB(self, method, source_case):
        return ac.BeamformerGIB(
            freq_data=source_case.freq_data, n=2, alpha=1e-12, eps_perc=2.0,
            steer=source_case.steer, method=method)

    @parametrize('method', ['LassoLars', 'LassoLarsBIC', 'OMPCV', 'NNLS',
        'fmin_l_bfgs_b', 'Split_Bregman', 'FISTA'])
    def case_BeamformerCMF(self, method, source_case):
        return ac.BeamformerCMF(
            freq_data=source_case.freq_data, max_iter=10,
            steer=source_case.steer, method=method)

    def case_BeamformerSODIX(self, source_case):
        return ac.BeamformerSODIX(
            freq_data=source_case.freq_data,
            steer=source_case.steer, max_iter=10)

    def case_BeamformerGridlessOrth(self, source_case):
        return ac.BeamformerGridlessOrth(
            freq_data=source_case.freq_data, r_diag=False,
            steer=source_case.steer, n=1, shgo={'n': 16})

    @parametrize('r_diag', [True, False])
    @parametrize('bf', get_beamformer_classes())
    def case_default(self, bf, r_diag, source_case):
        if r_diag and bf in [ac.BeamformerCapon, ac.BeamformerMusic, ac.BeamformerFunctional]:
            pytest.skip()
        return bf(
            freq_data=source_case.freq_data,
            steer=source_case.steer,
            r_diag=r_diag)

