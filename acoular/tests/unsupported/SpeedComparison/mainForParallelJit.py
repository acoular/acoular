#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:06:46 2017

@author: tomgensch

This script was for getting to know the mechanisms of the new prange function
of jit (numba version 0.34). As it turns out the automatic parallelization is
as faulty as in cython (same errors).
"""

import numpy as np
from numba import njit, float64, complex128, prange

nMics = 4
nGridPoints = 5
nFreqs = 1

csm = np.random.rand(nFreqs, nMics, nMics) + 1j*np.random.rand(nFreqs, nMics, nMics)
for cntFreqs in range(nFreqs):
    csm[cntFreqs, :, :] += csm[cntFreqs, :, :].T.conj()  # zu Hermitischer Matrix machen
r0 = np.random.rand(nGridPoints) #abstand aufpunkte-arraymittelpunkt
rm = np.random.rand(nGridPoints, nMics) #abstand aufpunkte-arraymikrofone
kj = np.zeros(nFreqs) + 1j*np.random.rand(nFreqs) 

@njit(float64[:,:](complex128[:,:,:], float64[:], float64[:,:], complex128[:]), parallel=True)
def loops_NumbaJit_parallelSlow(csm, r0, rm, kj):
    nFreqs = csm.shape[0]
    nGridPoints = len(r0)
    nMics = csm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in prange(nGridPoints):
            r01 = r0[cntGrid]
            rs = r01 ** 2
            
            temp1 = 0.0
            for cntMics in xrange(nMics): 
                temp2 = 0.0
                rm1 = rm[cntGrid, cntMics]
                temp3 = np.float32(kjj * (rm1 - r01))
                steerVec = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
                
                for cntMics2 in xrange(cntMics):
                    rm1 = rm[cntGrid, cntMics2]
                    temp3 = np.float32(kjj * (rm1 - r01))
                    steerVec1 = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec1
                temp1 += 2 * (temp2 * steerVec.conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * steerVec.conjugate() * steerVec).real
            
            beamformOutput[cntFreqs, cntGrid] = (temp1 / rs).real
    return beamformOutput

@njit(float64[:,:](complex128[:,:,:], float64[:], float64[:,:], complex128[:]), parallel=True)
def loops_NumbaJit_parallelFast(csm, r0, rm, kj):
    """ This method implements the prange over the Gridpoints, which is a direct
    implementation of the currently used c++ methods created with scipy.wave.
    
    Very strange: Just like with Cython, this implementation (prange over Gridpoints)
    produces wrong results. If one doesn't parallelize -> everything is good 
    (just like with Cython). Maybe Cython and Numba.jit use the same interpreter 
    to generate OpenMP-parallelizable code.
    
    BUT: If one uncomments the 'steerVec' declaration in the prange-loop over the
    gridpoints an error occurs. After commenting the line again and executing
    the script once more, THE BEAMFORMER-RESULTS ARE CORRECT (for repeated tries). 
    Funny enough the method is now twice as slow in comparison to the 
    'wrong version' (before invoking the error).
    """
    # init
    nFreqs = csm.shape[0]
    nGridPoints = len(r0)
    nMics = csm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    steerVec = np.zeros((nMics), np.complex128)
    
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in prange(nGridPoints):
#            steerVec = np.zeros((nMics), np.complex128)  # This is the line that has to be uncommented (see this methods documentation comment)
            rs = 0
            r01 = r0[cntGrid]
            
            for cntMics in xrange(nMics):
                rm1 = rm[cntGrid, cntMics]
                rs += 1.0 / (rm1**2)
                temp3 = np.float32(kjj * (rm1 - r01))
                steerVec[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
            rs = r01 ** 2
            
            temp1 = 0.0
            for cntMics in xrange(nMics):
                temp2 = 0.0
                for cntMics2 in xrange(cntMics):
                    temp2 = temp2 + csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]
                temp1 = temp1 + 2 * (temp2 * steerVec[cntMics].conjugate()).real
                temp1 = temp1 + (csm[cntFreqs, cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
            
            beamformOutput[cntFreqs, cntGrid] = (temp1 / rs).real
    return beamformOutput

beamformOutputFast = loops_NumbaJit_parallelFast(csm, r0, rm, kj)
beamformOutputSlow = loops_NumbaJit_parallelSlow(csm, r0, rm, kj)

#%% Referenz
beamformOutputRef = np.zeros((nFreqs, nGridPoints), np.float64)
e1 = np.zeros((nMics), np.complex128)

for cntFreqs in xrange(nFreqs):
    kjj = kj[cntFreqs].imag
    for cntGrid in xrange(nGridPoints):
        rs = 0
        r01 = r0[cntGrid]
        
        for cntMics in xrange(nMics):
            rm1 = rm[cntGrid, cntMics]
            rs += 1.0 / (rm1**2)
            temp3 = np.float32(kjj * (rm1 - r01)) 
            e1[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
        rs = r01 ** 2
        
        temp1 = 0.0
        for cntMics in xrange(nMics):
            temp2 = 0.0
            for cntMics2 in xrange(cntMics):
                temp2 += csm[cntFreqs, cntMics2, cntMics] * e1[cntMics2]
            temp1 += 2 * (temp2 * e1[cntMics].conjugate()).real
            temp1 += (csm[cntFreqs, cntMics, cntMics] * np.conjugate(e1[cntMics]) * e1[cntMics]).real
        
        beamformOutputRef[cntFreqs, cntGrid] = (temp1 / rs).real

diffFast = np.amax(abs(beamformOutputFast - beamformOutputRef), axis=0)
diffFast= np.amax(abs(diffFast), axis=0)

diffSlow = np.amax(abs(beamformOutputSlow - beamformOutputRef), axis=0)
diffSlow= np.amax(abs(diffSlow), axis=0)

print(diffFast)
print(diffSlow)