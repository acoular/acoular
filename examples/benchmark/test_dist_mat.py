# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

from numpy import linspace, float64, float32
import acoular

# assume 64 microphones and 2048 grid points
# provide both 32 and 64 bit input data

g32 = linspace(1, 2, 3*2048, dtype=float32).reshape((3,2048)) 
m32 = linspace(2, 3, 192, dtype=float32).reshape((3,64))
g64 = g32.astype(float64)
m64 = m32.astype(float64)

def test_dist_mat32():
    acoular.environments.dist_mat(g32,m32)

def test_dist_mat64():
    acoular.environments.dist_mat(g64,m64)
