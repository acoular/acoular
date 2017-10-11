#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
import time as tm

import numpy as np
from numba import guvectorize, complex128, float64

import sharedFunctions as shFncs
from beamformer import r_beamfull_os_classic, r_beamdiag_os_classic  # The benchmark (created with scipy.weave)
#from beamformer_withoutMP import r_beamfull_inverse_withoutMP  # also created with scipy.weave, but WITHOUT using multiple cores via OpenMP


#%% NUMBA - GUVECTORIZE
def csmBeamformer(csm, r0, rm, kj):
    """ Benchmark for comparison of correctness.
    """
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = np.exp(-1j * np.float32(kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])))
            # correct
            result = np.dot(steeringVector, csm[cntFreqs, :, :])
            beamformOutput[cntFreqs, cntGrid] = np.vdot(steeringVector, result)  
            
            # incorrect
#            result = np.dot(csm[cntFreqs, :, :], steeringVector)
#            beamformOutput[cntFreqs, cntGrid] = np.vdot(steeringVector, result)
    return beamformOutput.real / nMics**2


def correctResult(r0, rm, kj, eigVal, eigVec):
    nFreqs, nGridPoints, nMics = kj.shape[0], rm.shape[0], rm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = np.exp(-1j * np.float32(kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])))
            result = 0.0
            for cntEWs in range(eigVal.shape[1]):
#                spectralCSM = np.outer(eigVec[cntFreqs, :, cntEWs], eigVec[cntFreqs, :, cntEWs].conj())
#                result += np.inner(np.inner(steeringVector.conj(), spectralCSM), steeringVector) * eigVal[cntFreqs, cntEWs]
#            beamformOutput[cntFreqs, cntGrid] = result
                result += np.vdot(eigVec[cntFreqs, :, cntEWs], steeringVector.conj()) * eigVal[cntFreqs, cntEWs] * eigVec[cntFreqs, :, cntEWs]
            beamformOutput[cntFreqs, cntGrid] = np.dot(steeringVector, result)
    return beamformOutput.real / nMics**2

def loops_NumbaGuvectorizeOverGrid(r0, rm, kj, eigVal, eigVec):
    nFreqs, nGridPoints, nMics = kj.shape[0], rm.shape[0], rm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
        loops_Core(eigVal[cntFreqs, :], eigVec[cntFreqs, :, :], r0, rm, kj[cntFreqs].imag, nMics, result)
        beamformOutput[cntFreqs, :] = result
    return beamformOutput

@guvectorize([(float64[:], complex128[:,:], float64[:], float64[:], float64[:], float64[:], float64[:])], 
              '(e),(m,e),(),(m),(),()->()', nopython=True, target='parallel')
def loops_Core(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, 
                                                             waveNumber, nMicsForNormalization, result):
    # init
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operaions -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        temp = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distGridToArrayCenter[0]))
        steerVec[cntMics] = (np.cos(temp) - 1j * np.sin(temp))
    
    # performing matrix-vector-multiplication via spectral decomposition of the hermitian CSM-Matrix
    temp1 = 0.0
    for cntEigVal in range(len(eigVal)):
        temp2 = 0.0
        for cntMics in range(nMics):  # Algebraic multiplicity = Geometric multiplicity for any hermitian matrix
            temp2 += eigVec[cntMics, cntEigVal] * steerVec[cntMics]
        temp1 += (temp2 * temp2.conjugate() * eigVal[cntEigVal]).real
    result[0] = temp1 / nMicsForNormalization[0] ** 2


def loopsCascade_NumbaGuvectorizeOverGrid(r0, rm, kj, eigVal, eigVec):
    nFreqs, nGridPoints, nMics = kj.shape[0], rm.shape[0], rm.shape[1]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
        loopsCascade_Core(eigVal[cntFreqs, :], eigVec[cntFreqs, :, :], r0, rm, kj[cntFreqs].imag, nMics, result)
        beamformOutput[cntFreqs, :] = result
    return beamformOutput

@guvectorize([(float64[:], complex128[:,:], float64[:], float64[:], float64[:], float64[:], float64[:])], 
              '(e),(m,e),(),(m),(),()->()', nopython=True, target='parallel')
def loopsCascade_Core(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, 
                                                             waveNumber, nMicsForNormalization, result):
    # init
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operaions -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        temp = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distGridToArrayCenter[0]))
        steerVec[cntMics] = (np.cos(temp) - 1j * np.sin(temp))
    
    # performing matrix-vector-multiplication via spectral decomposition of the hermitian CSM-Matrix
    temp1 = 0.0
    for cntEigVal in range(len(eigVal)):
        temp2 = np.dot(eigVec[:, cntEigVal], steerVec)
        temp1 += (temp2 * temp2.conjugate() * eigVal[cntEigVal]).real
    result[0] = temp1 / nMicsForNormalization[0] ** 2
#%%  MAIN
listOfMics = [64, 100, 250, 500, 700, 1000]
listGridPoints = [100, 5000, 10000]  # Standard value: 12000  # The number of gridpoints doesn't seeme to have to great of an influence
nTrials = 10
listOfNFreqs = [10000]

#==============================================================================
# The benchmark function 'r_beamfull_inverse' and also other implementations of 
# the beamformer create a lot of overhead, which influences the computational 
# effort of the succeding function. This is mostly the case, if concurrent
# calculations are done (multiple cores). So often the first trial of a new 
# function takes some time longer than the other trials.
#==============================================================================

#funcsToTrial = [csmBeamformer, correctResult, r_beamfull_os_classic]  # full csm
#funcsToTrial = [csmBeamformer, correctResult, r_beamdiag_os_classic]  # removed diagonal of csm
funcsToTrial = [loopsCascade_NumbaGuvectorizeOverGrid, loops_NumbaGuvectorizeOverGrid, r_beamfull_os_classic] 
removeDiagOfCSM = False

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
            eigVal, eigVec = np.linalg.eigh(csm)
            indLow = 5  # 5
            indHigh = 10# 10
            
            # remove diagonal of csm
            if removeDiagOfCSM:
                for cntMics in range(nMics):
                    csm[cntFreqs, cntMics, cntMics] = 0
                eigValDiag, eigVecDiag = np.linalg.eigh(csm)
                refFunc = 'r_beamdiag_os_classic'
                resultReference = correctResult(r0, rm, kj, eigValDiag[:, indLow : indHigh], eigVecDiag[:, :, indLow : indHigh])  # For relative/absolute error
            else:
                refFunc = 'r_beamfull_os_classic'
                resultReference = correctResult(r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh])  # For relative/absolute error
            
            nameOfFuncsToTrial = map(lambda x: x.__name__, funcsToTrial)
            nameOfFuncsForError = [funcName for funcName in nameOfFuncsToTrial if funcName != refFunc]
            maxRelativeDeviation = np.zeros((len(funcsToTrial), nTrials))
            maxAbsoluteDeviation = np.zeros((len(funcsToTrial), nTrials))
            timeConsumption = [[] for _ in range(len(funcsToTrial))]
            indOfBaselineFnc = nameOfFuncsToTrial.index(refFunc)
            
            # Testing
            print(10*'-' + 'Testing of functions' + 10*'-')
            cntFunc = 0
            for func in funcsToTrial:
                print(func.__name__)
                for cntTrials in xrange(nTrials):
                    h = np.zeros((nFreqs, nGridPoints))
                    if func.__name__ == refFunc:
                        t0 = tm.time()
                        func(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
                        t1 = tm.time()
                        result = h / nMics**2
                    elif func.__name__ == 'correctResult' and removeDiagOfCSM:
                        t0 = tm.time()
                        output = func(r0, rm, kj, eigValDiag[:, indLow : indHigh], eigVecDiag[:, :, indLow : indHigh])
                        t1 = tm.time()
                        result = output
                    elif func.__name__ == 'csmBeamformer':
                        t0 = tm.time()
                        output = func(csm, r0, rm, kj)
                        t1 = tm.time()
                        result = output
                    else:
                        t0 = tm.time()
                        output = func(r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh])
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
                         nMics, nGridPoints, nFreqs, factorTimeConsump, maxAbsoluteDeviation, 0)  
            stringParameters = 'OvernightTestcasesBeamformer_nMics%s_nGridPoints%s_nFreqs%s_nTrials%s' %(nMics, nGridPoints, nFreqs, nTrials)
            
#            stringSaveName = 'Peter'
            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/cascadingSums/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/AllImportantMethods/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/EinflussGridpoints/AMDFX6100/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/JitPrange/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/Multithreading_02Threads/' + stringParameters

            shFncs.savingTimeConsumption(stringSaveName, saveTupel)  # saving as "stringSaveName.sav"
            
#            shFncs.plottingOfOvernightTestcasesBeamformer(stringSaveName + '.sav')  # plot of the current test-config

#==============================================================================
#The Following use of the numba decorators could lead to less code (as a function 
#body could be used more often) but is also slower, which is why it wasn't used
#in this comparison.
# signature = complex128[:,:](complex128[:,:,:], float64[:], float64[:,:])
# numbaOptimizedFunction= jit(signature, nopython=True)(plainPythonFunction.py_func)
#==============================================================================
