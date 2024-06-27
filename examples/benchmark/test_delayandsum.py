# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

import acoular
import numpy.random as random
from numpy import empty, float32, int32

# set_num_threads(1)

# set up test case
NUMMICS = 64
NUMSAMPLES = 2048
GRIDSIZE = 2048
NUM = 512

data = random.normal(size=(NUMSAMPLES, NUMMICS))
offsets = random.randint(low=0, high=NUMSAMPLES / 2, size=(GRIDSIZE, NUMMICS))
ifactor2 = random.uniform(low=0, high=1.0, size=(GRIDSIZE, NUMMICS))
steeramp = random.uniform(low=0, high=1.0, size=(GRIDSIZE, NUMMICS))
offsets5 = random.randint(low=0, high=NUMSAMPLES / 2, size=(NUM, GRIDSIZE, NUMMICS))
ifactor25 = random.uniform(low=0, high=1.0, size=(NUM, GRIDSIZE, NUMMICS))
steeramp5 = random.uniform(low=0, high=1.0, size=(NUM, GRIDSIZE, NUMMICS))
out = empty((NUM, GRIDSIZE))
autopower = empty((NUM, GRIDSIZE))


def test_delayandsum4():
    acoular.tfastfuncs._delayandsum4(data, offsets, ifactor2, steeramp, out, autopower)


def test_delayandsum5():
    acoular.tfastfuncs._delayandsum5(
        data,
        offsets5,
        ifactor25,
        steeramp5,
        out,
        autopower,
    )


data32 = data.astype(float32)
offsets532 = offsets5.astype(int32)
ifactor2532 = ifactor25.astype(float32)
steeramp532 = steeramp5.astype(float32)
out32 = out.astype(float32)
autopower32 = autopower.astype(float32)


def test_delayandsum532():
    acoular.tfastfuncs._delayandsum5(
        data32,
        offsets532,
        ifactor2532,
        steeramp532,
        out32,
        autopower32,
    )


rm = ifactor25.copy()
delays = ifactor25.copy()
interp = ifactor25.copy()
index = offsets5.copy()


def test_delays():
    acoular.tfastfuncs._delays(
        rm,
        343.0,
        interp,
        index,
    )


rm32 = rm.astype(float32)
interp32 = interp.astype(float32)
index32 = index.astype(int32)


def test_delays32():
    acoular.tfastfuncs._delays(
        rm32,
        float32(343.0),
        interp32,
        index32,
    )


r0 = random.uniform(low=0, high=1.0, size=(NUM, GRIDSIZE))


def test_steer_I():  # noqa: N802
    acoular.tfastfuncs._steer_I(
        rm,
        r0,
        steeramp5,
    )


def test_steer_II():  # noqa: N802
    acoular.tfastfuncs._steer_II(
        rm,
        r0,
        steeramp5,
    )


def test_steer_III():  # noqa: N802
    acoular.tfastfuncs._steer_III(
        rm,
        r0,
        steeramp5,
    )


def test_steer_IV():  # noqa: N802
    acoular.tfastfuncs._steer_IV(
        rm,
        r0,
        steeramp5,
    )
