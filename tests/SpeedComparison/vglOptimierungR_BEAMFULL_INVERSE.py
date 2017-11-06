#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Comparison of different optimization approaches to the 'r_beamfull_inverse' method.
Compared are: Numpy (matrix-vector-calculations), Numba and Cython. 

1. Background: 
Currently (python=2) performance-critical acoular methods (e.g.
faverage, all freqency-domain-beamformers, ...) are optimized via Scipy.weave
which translates code to c++, including compiling. Those executables can then
be imported in python.
Scipy.weave isn't supported anymore in python=3. Furthermore the executables of
already with Scipy.weave (python=2) compiled code cannot be imported in python=3.

2. Structure of comparison:
The benchmark in both errors and time consumption is always the Scipy.weave build, 
OpenMP optimized, c++ compiled code. Especially the relative and absolute inf-norm 
errors in the plots refer to the outputs of this function.

3. Remarks to the code:
In various codes below there are repeating patterns which may need some explanation:
a. There is a cast from 64-bit double precision to 32-bit precision of 'temp3',
    which is the argument to the exp() when calculating the steering vectors 
    -> This down-cast shortens the series expansion of exp() drastically which 
    leads to faster calculations while having acceptable errors. In fact if 
    there is no down-cast, the relative error between otherwise identical 
    methods is about 10^-8.
b. The exp() (when calculating the steering vector, see a.) is mostly replaced
    by a direct calculating of 'cos() - 1j*sin()', which can be done because
    the input 'temp3' of exp(temp3) is pure imaginary. Because of this the 
    calculation of exp(0)=1 in 'exp(0 - 1j*a) = exp(0) * (cos(a) - 1j*sin(a))'
    can be spared. This leads to further speed improvements.

4. Remark on the use of Cython:
See file 'cythonBeamformer.pyx' for remarks on Cython. It showed that at the 
moment Cython doesn't work to well for the beamformer case.

5. Others:


Versions used in this script:
numba=0.34.0
python=2.7.13



  # multiplizieren mit nMics erfolgt ausserhalb, sonst hier



"""
import time as tm
import threading
import gc

import numpy as np
from numba import jit, guvectorize, complex128, complex64, float64, float32, void, uint64, njit, prange

import sharedFunctions as shFncs
from cythonBeamformer import beamformerCython, beamformerCythonNOTparallel  # created with cython
from beamformer import r_beamfull_inverse  # The benchmark (created with scipy.weave)
from beamformer_withoutMP import r_beamfull_inverse_OhneMP  # also created with scipy.weave, but WITHOUT using multiple cores via OpenMP

#%% Formulate the Beamformer as VECTOR * MATRIX * VECTOR product
def vectorized(csm, e, h, r0, rm, kj):
    """ Uses Numpys fast array operations, distributed via the mkl-package.
    Those oparations are already optimized and use all available physical cores.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] \
                             * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            beamformOutput[cntFreqs, cntGrid] = np.inner(np.inner(steeringVector.T.conj(), csm[cntFreqs, :, :]), steeringVector)
    return beamformOutput.real

def vectorizedOptimized(csm, e, h, r0, rm, kj):
    """ Same as 'vectorized' but including both 3.a. & 3.b. of the documentation
    string at the beginning of this file. In opposite to the numba-optimized
    methods below, the use of 'cos() - 1j*sin()' instead of 'exp()' (see 3.a.)
    doesn't seem to have any speed improvement here.    
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in xrange(nGridPoints):
            temp3 = np.float32(kjj * (rm[cntGrid, :] - r0[cntGrid]))
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * (np.cos(temp3) - 1j * np.sin(temp3))
            beamformOutput[cntFreqs, cntGrid] = np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))
    return beamformOutput.real

@jit
def vectorized_NumbaJitOnly(csm, e, h, r0, rm, kj):
    """ Identical code to vectorized. Just decorated with the most basic 
    jit-optimization routine. If jit is able to translate all variables into
    primitive datatypes (NOT the native python objects) it will do that. If not,
    jit will fall back into 'Object mode' (native python objects) which will 
    mostly be much slower.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] \
                             * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            beamformOutput[cntFreqs, cntGrid] = np.inner(np.inner(steeringVector.T.conj(), csm[cntFreqs, :, :]), steeringVector)
    return beamformOutput.real

@jit(nopython=True)  # same as directly calling @njit
def vectorized_NumbaJit_nopythonTrue(csm, e, h, r0, rm, kj):
    """ In addition to 'vectorized_NumbaJitOnly' the nopython=True (or simply 
    @njit for numby>=0.34.) makes shure that if jit cannot translate the code 
    into primitive datatypes, it will NOT fall back into object mode but 
    instead returns an error.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            
#==============================================================================
#             #Not all numpy functions are supported in jit. It took some 
#             #tries to figure out how to implement the easy np.inner used in 
#             #'vectorized' into jit-supported functions.
#             

#             peer = np.inner(np.inner(steeringVector.T.conjugate(), csm[cntFreqs, :, :]), steeringVector)
#             peer2 = np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))  # Der scheint zu klappen 
#             peer3 = np.vdot(steeringVector, np.dot(csm[cntFreqs, :, :], steeringVector))
#             
#             versuch1 = np.inner(steeringVector, csm[cntFreqs, :, :])
#             versuch12 = np.inner(steeringVector.conjugate(), csm[cntFreqs, :, :])
#             versuch2 = np.dot(steeringVector, csm[cntFreqs, :, :])  # complex konj von versuch 12
#             versuch3 = np.dot(csm[cntFreqs, :, :], steeringVector)  # gleiche wie versuch1
#             versuch4 = np.dot(steeringVector.conjugate(), csm[cntFreqs, :, :])  # is das kompl conjugierte zu versuch1, versuch3
#             versuch5 = np.dot(csm[cntFreqs, :, :], steeringVector.conjugate())  # is das gleiche wie versuch12
#             ##--> Anscheinend ist die Syntax fuer x^H * A = dot(A, x.conj)
            beamformOutput[cntFreqs, cntGrid] = np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))  # This works
#==============================================================================
    return beamformOutput.real

@njit(float64[:,:](complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:]))
def vectorized_NumbaJit_nopythonTrue_DeclareInput(csm, e, h, r0, rm, kj):
    """ In addition to 'vectorized_NumbaJit_nopythonTrue' the in-/output of the
    method are declared in the decorator, which normally leads to speed 
    improvements (even though they're very little in this  particular  case).
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) 
            beamformOutput[cntFreqs, cntGrid] = np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :])).real
    return beamformOutput

@njit(float64[:,:](complex128[:,:,:], float64[:], float64[:,:], complex128[:]), parallel=True)
def vectorizedOptimized_NumbaJit_Parallel(csm, r0, rm, kj):
    """ The parallel=True flag turns on an automized parallezation process.
    When one wants to manually parallelize a certain loop one can do so by
    using prange instead of xrange/range. BUT in this method the prange
    produced errors. Maybe thats because the numpy package performs 
    parallelization itself, which is then in conflict with prange.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in xrange(nGridPoints):  # error when trying with prange
            temp3 = (kjj * (rm[cntGrid, :] - r0[cntGrid]))
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * (np.cos(temp3) - 1j * np.sin(temp3))
            beamformOutput[cntFreqs, cntGrid] = (np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))).real
    return beamformOutput

#%% Formulate the Beamformer as LOOPS
def loops_exactCopyOfCPP(csm, e, h, r0, rm, kj):
    """ A python copy of the current benchmark function, created with scipy.weave
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    nMics = csm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    steerVec = np.zeros((nMics), np.complex128)
    
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in xrange(nGridPoints):
            rs = 0
            r01 = r0[cntGrid]
            
            # Calculating of Steering-Vectors
            for cntMics in xrange(nMics):
                rm1 = rm[cntGrid, cntMics]
                rs += 1.0 / (rm1**2)
                temp3 = np.float32(kjj * (rm1 - r01))
                steerVec[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
            rs = r01 ** 2
            
            # Calculating of the matrix - vector - multiplication
            temp1 = 0.0
            for cntMics in xrange(nMics):
                temp2 = 0.0
                for cntMics2 in xrange(cntMics):
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]
                temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
            beamformOutput[cntFreqs, cntGrid] = temp1 / rs
    return beamformOutput

@njit(float64[:,:](complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:]))
def loops_NumbaJit_nopythonTrue_exactCopyOfCPP(csm, e, h, r0, rm, kj):
    """ See 'vectorized_NumbaJit_nopythonTrue_DeclareInput' for explenation of 
    the numba decorator.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    nMics = csm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    steerVec = np.zeros((nMics), np.complex128)
    
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in xrange(nGridPoints):
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
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]
                temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
            beamformOutput[cntFreqs, cntGrid] = temp1 / rs
    return beamformOutput


@njit(float64[:,:](complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:]), parallel=True)
def loops_NumbaJit_parallel_FirstWritingOfSteer(csm, e, h, r0, rm, kj):
    """ This method implements the parallelized loop (prange) over the 
    Gridpoints, which is a direct implementation of the currently used 
    c++ method created with scipy.wave.
    
    Very strange: Just like with Cython, this implementation (prange over Gridpoints)
    produces wrong results. If one doesn't parallelize -> everything is good 
    (just like with Cython). Maybe Cython and Numba.jit use the same interpreter 
    to generate OpenMP-parallelizable code.
    
    BUT: If one uncomments the 'steerVec' declaration in the prange-loop over the
    gridpoints an error occurs. After commenting the line again and executing
    the script once more, THE BEAMFORMER-RESULTS ARE CORRECT (for repeated tries). 
    Funny enough the method is now twice as slow in comparison to the 
    'wrong version' (before invoking the error).
    
    A workaround is given by 'loops_NumbaJit_parallel', which is much slower, 
    because the sterring vector is calculated redundantly.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
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
            
            beamformOutput[cntFreqs, cntGrid] = temp1 / rs
    return beamformOutput

@njit(float64[:,:](complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:]), parallel=True)
def loops_NumbaJit_parallel(csm, e, h, r0, rm, kj):
    """ Workaround for the prange error in jit. See documentation comment of
    'loops_NumbaJit_parallel_FirstWritingOfSteer'.
    For infos on the numba decorator see 'vectorizedOptimized_NumbaJit_Parallel'
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
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
                    steerVec1 = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1  # Steering vec is calculated redundantly--> very slow
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec1
                temp1 += 2 * (temp2 * steerVec.conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * steerVec.conjugate() * steerVec).real
            beamformOutput[cntFreqs, cntGrid] = temp1 / rs
    return beamformOutput

#%% Multithreading

#Due to pythons global interpreter lock (GIL) only one thread can run at a time.
#This means, that if one wants to make use of multiple cores, one has to release
#the GIL for concurrently running threads. Numbas jit can release the gil, if 
#all datatypes are primitive (nopython=True) via nogil=True.
#This doesn't have to be done with all the numba.guvectorized stuff, as the 
#multithreading happens automatically.
#I tested

# VERTIG MACHEN





nThreadsGlobal = 2  # einmal mit 2 und einmal mit 4 probieren.. vermutung: die saceh
# die numpy parallelsieirt (ohne jit) arbeitet eh auf beiden cores-> mehr threads bringt dann nichts meh

def vectorized_multiThreading(csm, e, h, r0, rm, kj):
    """ Prepares the Multithreading of 'vectorized_multiThreading_CoreFunction'.
    This method does not free the GIL. As descripted above (beginning of 
    Multithreading section) it therefore shouldn't run concurrently (on multiple 
    cores). BUT as numpys mkl package organizes concurrency itself (see 'vectorized'),
    this
    """
    nThreads = nThreadsGlobal
    dataSizePerThread = nGridPoints / nThreads
    startingIndexPerThread = [cnt * dataSizePerThread for cnt in range(nThreads + 1)]
    startingIndexPerThread[-1] = nGridPoints
    threads = [threading.Thread(target=vectorized_multiThreading_CoreFunction, args=(csm, e, h, r0, rm, kj, startingIndexPerThread[cnt], startingIndexPerThread[cnt+1])) for cnt in range(nThreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return h
def vectorized_multiThreading_CoreFunction(csm, e, h, r0, rm, kj, startPoint, endPoint):
    nFreqs = csm.shape[0]
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(startPoint, endPoint):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            h[cntFreqs, cntGrid] = (np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))).real


def vectorized_NumbaJit_multiThreading(csm, e, h, r0, rm, kj):
    """ Prepares the Multithreading of 'vectorized_NumbaJit_multiThreading_CoreFunction'
    """
    nThreads = nThreadsGlobal
    dataSizePerThread = nGridPoints / nThreads
    startingIndexPerThread = [cnt * dataSizePerThread for cnt in range(nThreads + 1)]
    startingIndexPerThread[-1] = nGridPoints
    threads = [threading.Thread(target=vectorized_NumbaJit_multiThreading_CoreFunction, args=(csm, e, h, r0, rm, kj, startingIndexPerThread[cnt], startingIndexPerThread[cnt+1])) for cnt in range(nThreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return h
@njit(void(complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:], uint64, uint64), nogil=True)
def vectorized_NumbaJit_multiThreading_CoreFunction(csm, e, h, r0, rm, kj, startPoint, endPoint):
    nFreqs = csm.shape[0]
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(startPoint, endPoint):
            steeringVector = rm[cntGrid, :] / r0[cntGrid] * np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            h[cntFreqs, cntGrid] = (np.vdot(steeringVector, np.dot(steeringVector, csm[cntFreqs, :, :]))).real


def loops_NumbaJit_multiThreading(csm, e, h, r0, rm, kj):
    """ Prepares the Multithreading of 'loops_NumbaJit_multiThreading_CoreFunction'.
    Here the cores are used as they should which means: 
    You spawn 2 threads -> cpu uses 2 cores,
    You spawn 3 threads -> cpu uses 3 cores...
    """
    nThreads = nThreadsGlobal
    dataSizePerThread = nGridPoints / nThreads
    startingIndexPerThread = [cnt * dataSizePerThread for cnt in range(nThreads + 1)]
    startingIndexPerThread[-1] = nGridPoints
    threads = [threading.Thread(target=loops_NumbaJit_multiThreading_CoreFunction, args=(csm, e, h, r0, rm, kj, startingIndexPerThread[cnt], startingIndexPerThread[cnt+1])) for cnt in range(nThreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return h
@njit(void(complex128[:,:,:], complex128[:], float64[:,:], float64[:], float64[:,:], complex128[:], uint64, uint64), nogil=True)
def loops_NumbaJit_multiThreading_CoreFunction(csm, e, h, r0, rm, kj, startPoint, endPoint):
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    steerVec = np.zeros((nMics), np.complex128)
    
    for cntFreqs in xrange(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in xrange(startPoint, endPoint):
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
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]
                temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
            h[cntFreqs, cntGrid] = temp1 / rs

#%% NUMBA - GUVECTORIZE

@guvectorize([void(complex128[:,:], float64[:], float64[:,:], complex128[:], float64[:])], '(m,m),(g),(g,m),()->(g)', nopython=True, target='parallel')
def loops_NumbaGuvectorize(csm, r0, rm, kj, h):
    """ Creating a Numpy-Ufunc: Define it for an input which is n-dimensional.
    Then call it with an input which is n+1 dimensional. Python takes care of 
    the parallelization over the available cores itself.
    In this case python parallelizes over the frequencies.
    Numbas guvectorize doesn't return a value but has to overwrite an result 
    vector passed to the method (in this case 'h') as the last input.
    
    Short description of the guvectorize decorator:
        1. Input-Argument: Declaration of output/input datatypes just like
            with jit, but with a obligatary [] around it
        2. '(m,m),(g)...': A symbolic explenation of the input dimensions. In this
            case 'loops_NumbaGuvectorize' is defined for the following input-dim 
            (csm[nMics x nMics], r0[nGridpoints], rm[nGridpoints x nMics], kj (a scalar), h[nGridpoints])
            , where 'h' contains the calculated results (identified by '->').
            When you then give an input which tensorical order is exactly one order 
            higher then the here made definition (e.g. csm[!nFreqs! x nMics x nMics]), 
            numba automatically distributes the new tensor order onto the 
            muliple cores (in our case every core computes the beamformer map 
            for a single frequency independently of the others)
        3. target: one can compute only on one core (target='CPU'), all available 
            cores (target='parallel') or even on graphic cards (target='cuda') (if drivers are installed)
        4. nopython: See jit-decorator, used above
        
    See also man page "http://numba.pydata.org/".
    
    REMARK: Strangly this seemed only to work, if the added order of CSM was its
        first dimension. E.g. csm[nMics x nMics x nFreqs] didn't seem to work.
    """
    nGridPoints = r0.shape[0]
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)
    
    kjj = kj[0].imag  # If input is scalar, it has to be dereferenced using the 'variable[0]'-syntax
    for cntGrid in xrange(nGridPoints):
        rs = 0.0
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
                temp2 += csm[cntMics2, cntMics] * steerVec[cntMics2]
            temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
            temp1 += (csm[cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
        h[cntGrid] = temp1.real / rs

#@njit(float64[:,:](complex128[:,:,:], float64[:], float64[:,:], complex128[:]))  # right now it doesn't seem to be supported for jit-optimized methods to call guvectorized subroutines. Maybe this will be changed in the future
def loops_NumbaGuvectorizeOverGrid(csm, r0, rm, kj):
    """ Similar to 'loops_NumbaGuvectorize', but in this case the UFunc parallelizes
    over the Gridpoints (as it is done in the scipy.weave version). This leads
    to significant speed improvements. 
    Thoughts on the speed improvements: I can't see why the pipelining should 
    work any more effective in comparison to 'loops_NumbaGuvectorize' (where
    the parallelization is done over the frequency-loop), as in both cases the 
    most time is spend in the loop over the gridpoints, so the chain of 
    instructions should essentially be the same.
    BUT in 'loops_NumbaGuvectorize' the slice of every currently calculated 
    frequency of the CSM is loaded into the shared Cache (e.g. with 4 cores a 
    '4 x nMics x nMics'-tensor is loaded into the shared Cache), whereas with 
    'loops_NumbaGuvectorizeOverGrid' only a '1 x nMics x nMics'-tensor is 
    loaded into the shared Cache. This maybe leads to better managing of resources.
    """
    nGridPoints = r0.shape[0]
    nFreqs = csm.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
        loops_NumbaGuvectorizeOverGrid_CoreFunction(csm[cntFreqs, :, :], r0, rm, kj[cntFreqs], result)
        beamformOutput[cntFreqs, :] = result
    return beamformOutput
    
@guvectorize([(complex128[:,:], float64[:], float64[:], complex128[:], float64[:])], 
              '(m,m),(),(m),()->()', nopython=True, target='parallel')
def loops_NumbaGuvectorizeOverGrid_CoreFunction(csm, r0, rm, kj, h):
    """ CoreFunction of 'loops_NumbaGuvectorizeOverGrid', which does the 
    parallelization over the gridpoints.
    """
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)
    kjj = kj[0].imag
    
    rs = 0.0
    r01 = r0[0]
    for cntMics in xrange(nMics):
        rm1 = rm[cntMics]
        rs += 1.0 / (rm1**2)
        temp3 = np.float32(kjj * (rm1 - r01))
#==============================================================================
        steerVec[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
#        steerVec[cntMics] = np.exp(-1j * temp3) * rm1  # is analytically the same as the last line
#
#          With exp(), instead of cos + 1j* sin, the function is noticeable slower 
#          AND the relative error is ca 10^-8 (as with those implementations which 
#          don't perform the down cast from double to 32-bit-precision) 
#          -> Maybe the exp() performs implicitly a cast back to double if its 
#          input is imaginary?!
#==============================================================================
    rs = r01 ** 2
    
    temp1 = 0.0
    for cntMics in xrange(nMics):
        temp2 = 0.0
        for cntMics2 in xrange(cntMics):
            temp2 += csm[cntMics2, cntMics] * steerVec[cntMics2]
        temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
        temp1 += (csm[cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
    h[0] = temp1 / rs


def loops_NumbaGuvectorizeOverGridNoCast(csm, r0, rm, kj):
    nGridPoints = r0.shape[0]
    nFreqs = csm.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
        loops_NumbaGuvectorizeOverGridNoCast_CoreFunction(csm[cntFreqs, :, :], r0, rm, kj[cntFreqs], result)
        beamformOutput[cntFreqs, :] = result
    return beamformOutput
    
@guvectorize([(complex128[:,:], float64[:], float64[:], complex128[:], float64[:])], 
              '(m,m),(),(m),()->()', nopython=True, target='parallel')
def loops_NumbaGuvectorizeOverGridNoCast_CoreFunction(csm, r0, rm, kj, h):
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)
    kjj = kj[0].imag
    
    rs = 0.0
    r01 = r0[0]
    for cntMics in xrange(nMics):
        rm1 = rm[cntMics]
        rs += 1.0 / (rm1**2)
        temp3 = kjj * (rm1 - r01)
        steerVec[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
    rs = r01 ** 2
    
    temp1 = 0.0
    for cntMics in xrange(nMics):
        temp2 = 0.0
        for cntMics2 in xrange(cntMics):
            temp2 += csm[cntMics2, cntMics] * steerVec[cntMics2]
        temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
        temp1 += (csm[cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
    h[0] = temp1 / rs


def loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit(csm, r0, rm, kj):
    nGridPoints = r0.shape[0]
    nFreqs = csm.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
#        loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit_CoreFunction(csm[cntFreqs, :, :], r0, rm, kj[cntFreqs], result)
        loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit_CoreFunction(np.complex64(csm[cntFreqs, :, :]), np.float32(r0), np.float32(rm), np.complex64(kj[cntFreqs]), np.float32(result))
        beamformOutput[cntFreqs, :] = result
    return beamformOutput
    
@guvectorize([(complex64[:,:], float32[:], float32[:], complex64[:], float32[:])], '(m,m),(),(m),()->()', nopython=True, target='parallel')
#@guvectorize([(complex128[:,:], float64[:], float64[:], complex128[:], float64[:])], '(m,m),(),(m),()->()', nopython=True, target='parallel')
def loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit_CoreFunction(csm, r0, rm, kj, h):
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex64)
    kjj = np.float32(kj[0].imag)
    
    r01 = np.float32(r0[0])
    for cntMics in xrange(nMics):
        rm1 = np.float32(rm[cntMics])
        temp3 = np.float32(kjj * (rm[cntMics] - r01))
        steerVec[cntMics] = np.complex64((np.cos(temp3) - 1j * np.sin(temp3)) * rm1)
    rs = r01 * r01

    temp1 = np.float32(0.0)
#    temp1 = np.float64(0.0)    
    for cntMics in xrange(nMics):
        temp2 = np.complex64(0.0 + 0.0j)
#        temp2 = np.complex128(0.0 + 0.0j)
        for cntMics2 in xrange(cntMics):
            temp2 += csm[cntMics2, cntMics] * steerVec[cntMics2]
        temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
        temp1 += (csm[cntMics, cntMics] * np.conjugate(steerVec[cntMics]) * steerVec[cntMics]).real
    h[0] = temp1 / rs
#%%  MAIN
listOfMics = [64] #[64, 100, 250, 500, 700, 1000]
listGridPoints = [5] # [500, 5000, 10000]  # Standard value: 12000  # The number of gridpoints doesn't seeme to have to great of an influence
nTrials = 10
listOfNFreqs = [20]

#==============================================================================
# The benchmark function 'r_beamfull_inverse' and also other implementations of 
# the beamformer create a lot of overhead, which influences the computational 
# effort of the succeding function. This is mostly the case, if concurrent
# calculations are done (multiple cores). So often the first trial of a new 
# function takes some time longer than the other trials.
#==============================================================================

#funcsToTrial = [vectorized, vectorizedOptimized, vectorized_NumbaJitOnly, \
#                vectorized_NumbaJit_nopythonTrue, vectorized_NumbaJit_nopythonTrue_DeclareInput, \
#                vectorizedOptimized_NumbaJit_Parallel, \
#                loops_exactCopyOfCPP, loops_NumbaJit_nopythonTrue_exactCopyOfCPP, \
#                loops_NumbaJit_parallel_FirstWritingOfSteer, loops_NumbaJit_parallel, \
#                vectorized_multiThreading, vectorized_NumbaJit_multiThreading, loops_NumbaJit_multiThreading, \
#                loops_NumbaGuvectorize, loops_NumbaGuvectorizeOverGrid, \
#                r_beamfull_inverse_OhneMP, r_beamfull_inverse]

#funcsToTrial = [vectorized, vectorizedOptimized, beamformerCythonNOTparallel, loops_NumbaJit_parallel_FirstWritingOfSteer, \
#                vectorized_multiThreading, vectorized_NumbaJit_multiThreading, loops_NumbaJit_multiThreading, \
#                loops_NumbaGuvectorize, loops_NumbaGuvectorizeOverGrid, \
#                r_beamfull_inverse_OhneMP, r_beamfull_inverse]

funcsToTrial = [loops_NumbaGuvectorize, loops_NumbaGuvectorizeOverGrid, loops_NumbaGuvectorizeOverGridNoCast, loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit, r_beamfull_inverse]

for nMics in listOfMics:
    for nGridPoints in listGridPoints:
        for nFreqs in listOfNFreqs:
            # Init
            print(10*'-' + 'New Test configuration: nMics=%s, nGridpoints=%s, nFreqs=%s' %(nMics, nGridPoints, nFreqs) + 10*'-')
            print(10*'-' + 'Creation of inputInputs' + 10*'-')
            
            # Inputs for the beamformer methods:
            # At the moment the beamformer-methods are called once per 
            # frequency (CSM is a Matrix, no 3rd-order-tensor)
            # For easier camparability we build the CSM as a 3rd-order-tensor) instead
            csm = np.random.rand(nFreqs, nMics, nMics) + 1j*np.random.rand(nFreqs, nMics, nMics)  # cross spectral matrix
            for cntFreqs in range(nFreqs):
                csm[cntFreqs, :, :] += csm[cntFreqs, :, :].T.conj()  # make CSM hermetical
            e = np.random.rand(nMics) + 1j*np.random.rand(nMics)  # has no usage
            h = np.zeros((nFreqs, nGridPoints))  # results are stored here, if function has no return value
            r0 = np.random.rand(nGridPoints)  # distance between gridpoints and middle of array
            rm = np.random.rand(nGridPoints, nMics)  # distance between gridpoints and all mics in the array
            kj = np.zeros(nFreqs) + 1j*np.random.rand(nFreqs) # complex wavenumber 
            
            nameOfFuncsToTrial = map(lambda x: x.__name__, funcsToTrial)
            nameOfFuncsForError = [funcName for funcName in nameOfFuncsToTrial if funcName != 'r_beamfull_inverse']
            maxRelativeDeviation = np.zeros((len(funcsToTrial), nTrials))
            maxAbsoluteDeviation = np.zeros((len(funcsToTrial), nTrials))
            timeConsumption = [[] for _ in range(len(funcsToTrial))]
            indOfBaselineFnc = nameOfFuncsToTrial.index('r_beamfull_inverse')
            
            print(10*'-' + 'Onetime calculation of "r_beamfull_inverse" for error reference' + 10*'-')
            r_beamfull_inverse(csm, e, h, r0, rm, kj)
            resultReference = h  # For relative/absolute error
            gc.collect()
            
            # Testing
            print(10*'-' + 'Testing of functions' + 10*'-')
            cntFunc = 0
            for func in funcsToTrial:
                print(func.__name__)
                for cntTrials in xrange(nTrials):
                    h = np.zeros((nFreqs, nGridPoints))
                    if func.__name__ == 'r_beamfull_inverse' or func.__name__ == 'r_beamfull_inverse_OhneMP':
                        t0 = tm.time()
                        func(csm, e, h, r0, rm, kj)
                        t1 = tm.time()
                        result = h
#                        gc.collect()
                    elif func.__name__ == 'loops_NumbaGuvectorize':
                        t0 = tm.time()
                        func(csm, r0, rm, kj, h)
                        t1 = tm.time()
                        result = h
                    elif func.__name__ == 'loops_NumbaGuvectorizeOverGrid' or func.__name__ == 'vectorizedOptimized_NumbaJit_Parallel' or func.__name__ == 'loops_NumbaGuvectorizeOverGridNoCast' or func.__name__ == 'loops_NumbaGuvectorizeOverGridAllCalcsIn32Bit':
                        t0 = tm.time()
                        output = func(csm, r0, rm, kj)
                        t1 = tm.time()
                        result = output
                    elif func.__name__ == 'beamformerCython' or func.__name__ == 'beamformerCythonNOTparallel':
                        t0 = tm.time()
                        output = func(csm, r0, rm, kj)
                        t1 = tm.time()
                        result = np.array(output)
                    else:
                        t0 = tm.time()
                        output = func(csm, e, h, r0, rm, kj)
                        t1 = tm.time()
                        result = output
                    timeConsumption[cntFunc].append(t1 - t0)
                    relativeDiffBetweenNewCodeAndRef = (result - resultReference) / (result + resultReference) * 2  # error in relation to the resulting value
                    maxRelativeDeviation[cntFunc, cntTrials] = np.amax(np.amax(abs(relativeDiffBetweenNewCodeAndRef), axis=1), axis=0)  # relative error in inf-norm
                    maxAbsoluteDeviation[cntFunc, cntTrials] = np.amax(np.amax(abs(result - resultReference), axis=1), axis=0)  # absolute error in inf-norm
                cntFunc += 1
            factorTimeConsump = [np.mean(timeConsumption[cnt]) for cnt in range(0, len(funcsToTrial))] \
                                / np.mean(timeConsumption[indOfBaselineFnc])
            
            # Save the current test-config as .sav
            helpString = 'The order of the variables is: \n nameOfFuncsToTrial \n maxRelativeDeviation'\
                '\n timeConsumption [nFuncs, nTrials] \n nMics \n nGridPoints \n nFreqs '\
                '\n Factor of time consumption (in relation to the original .cpp) \n maxAbsoluteDeviation \n nThreadsGlobal'
            saveTupel = (helpString, nameOfFuncsToTrial, maxRelativeDeviation, timeConsumption, 
                         nMics, nGridPoints, nFreqs, factorTimeConsump, maxAbsoluteDeviation, nThreadsGlobal)
            stringParameters = 'OvernightTestcasesBeamformer_nMics%s_nGridPoints%s_nFreqs%s_nTrials%s' %(nMics, nGridPoints, nFreqs, nTrials)
            
            stringSaveName = 'Peter'
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/AllImportantMethods/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/EinflussGridpoints/AMDFX6100/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/JitPrange/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/Multithreading_02Threads/' + stringParameters

            shFncs.savingTimeConsumption(stringSaveName, saveTupel)  # saving as "stringSaveName.sav"
            
            shFncs.plottingOfOvernightTestcasesBeamformer(stringSaveName + '.sav')  # plot of the current test-config

#==============================================================================
#The Following use of the numba decorators could lead to less code (as a function 
#body could be used more often) but is also slower, which is why it wasn't used
#in this comparison.
# signature = complex128[:,:](complex128[:,:,:], float64[:], float64[:,:])
# numbaOptimizedFunction= jit(signature, nopython=True)(plainPythonFunction.py_func)
#==============================================================================
