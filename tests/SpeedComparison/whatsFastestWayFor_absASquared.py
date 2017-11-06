#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:24:36 2017

@author: tomgensch
"""

import numba as nb
import time as tm

@nb.njit
def absASquared(N):
    a = 42.0 + 87.0j
    for cnt in xrange(N):
        peter = abs(a) * abs(a)
    return peter
@nb.njit
def aTimesAConj(N):
    a = 42.0 + 87.0j
    for cnt in xrange(N):
        peter = (a * a.conjugate()).real
    return peter
@nb.njit
def realSquaredPlusImagSquared(N):
    a = 42.0 + 87.0j
    for cnt in xrange(N):
        peter = a.real * a.real + a.imag * a.imag
    return peter

N = 3000000000
t0 = tm.time()
erg1 = absASquared(N)
timeAbsASquared = tm.time() - t0

t0 = tm.time()
erg2 = aTimesAConj(N)
timeATimesAConj = tm.time() - t0

t0 = tm.time()
erg3 = realSquaredPlusImagSquared(N)
timeRealSquaredPlusImagSquared = tm.time() - t0

faktorAbsSquareVstimeRealSquaredPlusImagSquared = timeAbsASquared / timeRealSquaredPlusImagSquared
faktortimeATimesAConjVstimeRealSquaredPlusImagSquared = timeATimesAConj / timeRealSquaredPlusImagSquared

print('abs(a)**2 / (a.real**2 + a.imag**2): %s' %(faktorAbsSquareVstimeRealSquaredPlusImagSquared))
print('a*a.conj / (a.real**2 + a.imag**2): %s' %(faktortimeATimesAConjVstimeRealSquaredPlusImagSquared))
