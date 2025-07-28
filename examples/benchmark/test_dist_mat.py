# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

import acoular as ac
import numpy as np

# assume 64 microphones and 2048 grid points
# provide both 32 and 64 bit input data

g32 = np.linspace(1, 2, 3 * 2048, dtype=np.float32).reshape((3, 2048))
m32 = np.linspace(2, 3, 192, dtype=np.float32).reshape((3, 64))
g64 = g32.astype(np.float64)
m64 = m32.astype(np.float64)


def test_dist_mat32():
    ac.environments.dist_mat(g32, m32)


def test_dist_mat64():
    ac.environments.dist_mat(g64, m64)
