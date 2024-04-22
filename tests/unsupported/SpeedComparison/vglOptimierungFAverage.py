#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
import time as tm
import gc

import numpy as np
from numba import njit, guvectorize, complex128, void, prange

import sharedFunctions as shFncs
from beamformer import faverage  # The benchmark (created with scipy.weave)
from beamformer_withoutMP import faverage_OhneMP

#%% Numba - njit
@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def loops_Njit(csm, SpecAllChn):
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in range(nFreqs):
        for cntRow in range(nMics):
            temp = np.conj(SpecAllChn[cntFreq, cntRow])
            for cntColumn in range(nMics):
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllChn[cntFreq, cntColumn]
    return csm

@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]), parallel=True)
def loops_Njit_Parallel(csm, SpecAllChn):
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in range(nFreqs):
        for cntRow in range(nMics):
            temp = np.conj(SpecAllChn[cntFreq, cntRow])
            for cntColumn in range(nMics):
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllChn[cntFreq, cntColumn]
    return csm

@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]), parallel=True)
def loops_Njit_Parallel_Prange(csm, SpecAllChn):
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in range(nFreqs):
        for cntRow in prange(nMics):
            temp = np.conj(SpecAllChn[cntFreq, cntRow])
            for cntColumn in range(nMics):
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllChn[cntFreq, cntColumn]
    return csm

#%% create CSM via complex transpose of lower triangular matrix
@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def loopsComplexTranspose_Numpy(csm, SpecAllChn):
    nFreqs = csm.shape[0]
    for cntFreq in range(nFreqs):
        csm[cntFreq, :, :] += np.outer(np.conj(SpecAllChn[cntFreq, :]), SpecAllChn[cntFreq, :])
    return csm

@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def loopsOnlyTriangularMatrix_Njit(csm, SpecAllChn):
    """ one could only build the lower triangular csm and then, after averaging
    over all the ensenbles creating the whole csm by complex transposing.
    One could maybe use sparse CSR/CSC matrices (even though the CSM is not too big, so advantages are maybe small.)
    """
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in range(nFreqs):
        for cntRow in range(nMics):
            temp = np.conj(SpecAllChn[cntFreq, cntRow])
            for cntColumn in range(cntRow):  # only half of the operations in respect to 'loops_Njit'
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllChn[cntFreq, cntColumn]
    return csm

#%% Numba - guvectorize

# =============================================================================
# I don't think that parallelizing over the mics is in this case feasible.
# At least i can't think of a way to abstract the faverage procedure on one level below.
# It is however feasible to parallelize over the frequencies
# =============================================================================

@guvectorize([void(complex128[:,:], complex128[:], complex128[:,:])], '(m,m),(m)->(m,m)',
              nopython=True, target='cpu')
def loops_GuvectorizeOverFreqs_singleThreadedCPU(csm, SpecAllChn, result):
    nMics = csm.shape[0]
    for cntRow in range(nMics):
        temp = np.conj(SpecAllChn[cntRow])
        for cntColumn in range(nMics):
            result[cntRow, cntColumn] = csm[cntRow, cntColumn] + temp * SpecAllChn[cntColumn]

@guvectorize([void(complex128[:,:], complex128[:], complex128[:,:])], '(m,m),(m)->(m,m)',
              nopython=True, target='parallel')
def loops_GuvectorizeOverFreqs_multiThreadedCPU(csm, SpecAllChn, result):
    nMics = csm.shape[0]
    for cntRow in range(nMics):
        temp = np.conj(SpecAllChn[cntRow])
        for cntColumn in range(nMics):
            result[cntRow, cntColumn] = csm[cntRow, cntColumn] + temp * SpecAllChn[cntColumn]

@guvectorize([void(complex128[:,:], complex128[:], complex128[:,:])], '(m,m),(m)->(m,m)',
              nopython=True, target='cpu')
def loopsOnlyTriangularMatrix_GuvectorizeOverFreqs_singleThreadedCPU(csm, SpecAllChn, result):
    nMics = csm.shape[0]
    for cntRow in range(nMics):
        temp = np.conj(SpecAllChn[cntRow])
        for cntColumn in range(cntRow):
            result[cntRow, cntColumn] = csm[cntRow, cntColumn] + temp * SpecAllChn[cntColumn]

@guvectorize([void(complex128[:,:], complex128[:], complex128[:,:])], '(m,m),(m)->(m,m)',
              nopython=True, target='parallel')
def loopsOnlyTriangularMatrix_GuvectorizeOverFreqs_multiThreadedCPU(csm, SpecAllChn, result):
    nMics = csm.shape[0]
    for cntRow in range(nMics):
        temp = np.conj(SpecAllChn[cntRow])
        for cntColumn in range(cntRow):
            result[cntRow, cntColumn] = csm[cntRow, cntColumn] + temp * SpecAllChn[cntColumn]

#%%  MAIN
listOfMics = [500, 700, 1000]  # default: 64
listOfNFreqs = [2**cnt for cnt in range(4, 11)]  # default: 2048
nTrials = 10


#==============================================================================
# The benchmark function 'faverage' and also other implementations of
# the beamformer create a lot of overhead, which influences the computational
# effort of the succeding function. This is mostly the case, if concurrent
# calculations are done (multiple cores). So often the first trial of a new
# function takes some time longer than the other trials.
#==============================================================================

#funcsToTrial = [loopsComplexTranspose_Numpy, loops_Njit, loops_Njit_Parallel, loops_Njit_Parallel_Prange,
#                loopsOnlyTriangularMatrix_Njit, loops_GuvectorizeOverFreqs_singleThreadedCPU,
#                loops_GuvectorizeOverFreqs_multiThreadedCPU,
#                loopsOnlyTriangularMatrix_GuvectorizeOverFreqs_singleThreadedCPU,
#                loopsOnlyTriangularMatrix_GuvectorizeOverFreqs_multiThreadedCPU,
#                faverage_OhneMP, faverage]
funcsToTrial = [loopsOnlyTriangularMatrix_Njit, faverage]

for nMics in listOfMics:
    for nFreqs in listOfNFreqs:
        # Init
        print(10*'-' + 'New Test configuration: nMics=%s, nFreqs=%s' %(nMics, nFreqs) + 10*'-')
        print(10*'-' + 'Creation of inputInputs' + 10*'-')

        csm = np.zeros((nFreqs, nMics, nMics), np.complex128)
        spectrumInput = np.random.rand(nFreqs, nMics) + \
                1j*np.random.rand(nFreqs, nMics)

        nameOfFuncsToTrial = map(lambda x: x.__name__, funcsToTrial)
        nameOfFuncsForError = [funcName for funcName in nameOfFuncsToTrial if funcName != 'faverage']
        maxRelativeDeviation = np.zeros((len(funcsToTrial), nTrials))
        maxAbsoluteDeviation = np.zeros((len(funcsToTrial), nTrials))
        timeConsumption = [[] for _ in range(len(funcsToTrial))]
        indOfBaselineFnc = nameOfFuncsToTrial.index('faverage')

        print(10*'-' + 'Onetime calculation of "faverage" for error reference' + 10*'-')
        faverage(csm, spectrumInput)
        resultReference = csm  # For relative/absolute error
        gc.collect()

        # Testing
        print(10*'-' + 'Testing of functions' + 10*'-')
        cntFunc = 0
        for func in funcsToTrial:
            print(func.__name__)
            for cntTrials in xrange(nTrials):
                csm = np.zeros((nFreqs, nMics, nMics), np.complex128)
                resultHelp = np.zeros((nFreqs, nMics, nMics), np.complex128)
                if func.__name__ == 'faverage' or func.__name__ == 'faverage_OhneMP':
                    t0 = tm.time()
                    func(csm, spectrumInput)
                    t1 = tm.time()
                    result = csm
                elif func.__name__ == 'loops_GuvectorizeOverFreqs':
                    t0 = tm.time()
                    func(csm, spectrumInput, resultHelp)
                    t1 = tm.time()
                    result = resultHelp
                else:
                    t0 = tm.time()
                    output = func(csm, spectrumInput)
                    t1 = tm.time()
                    result = output
                timeConsumption[cntFunc].append(t1 - t0)
                relativeDiffBetweenNewCodeAndRef = (result - resultReference) / (result + resultReference) * 2  # error in relation to the resulting value
                maxRelativeDeviation[cntFunc, cntTrials] = np.amax(np.amax(np.amax(abs(relativeDiffBetweenNewCodeAndRef), axis=0), axis=0), axis=0) + 10.0**-20  # relative error in inf-norm
                maxAbsoluteDeviation[cntFunc, cntTrials] = np.amax(np.amax(np.amax(abs(result - resultReference), axis=0), axis=0), axis=0) + 10.0**-20  # absolute error in inf-norm
            cntFunc += 1
        factorTimeConsump = [np.mean(timeConsumption[cnt]) for cnt in range(0, len(funcsToTrial))] \
                            / np.mean(timeConsumption[indOfBaselineFnc])

        # Save the current test-config as .sav
        helpString = 'The order of the variables is: \n nameOfFuncsToTrial \n maxRelativeDeviation'\
                '\n timeConsumption [nFuncs, nTrials] \n nMics \n nGridPoints \n nFreqs '\
                '\n Factor of time consumption (in relation to the original .cpp) \n maxAbsoluteDeviation \n nThreadsGlobal'
        saveTupel = (helpString, nameOfFuncsToTrial, maxRelativeDeviation, timeConsumption,
                     nMics, 0, nFreqs, factorTimeConsump, maxAbsoluteDeviation, 0)
        stringParameters = 'faverage_TestcasesTimeConsumption_nMics%s_nFreqs%s_nTrials%s' %(nMics, nFreqs, nTrials)

#        stringSaveName = 'Peter'
        stringSaveName = 'Sicherung_DurchgelaufeneTests/faverage/' + stringParameters

        shFncs.savingTimeConsumption(stringSaveName, saveTupel)  # saving as "stringSaveName.sav"
#        shFncs.plottingOfOvernightTestcasesBeamformer(stringSaveName + '.sav')  # plot of the current test-config

