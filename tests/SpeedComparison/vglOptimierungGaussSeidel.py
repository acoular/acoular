#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checking various implementations of the damas solver


Versions used in this script:
numba=0.34.0
python=2.7.13
"""
import time as tm
import threading
import gc

import numpy as np
import numba as nb

import sharedFunctions as shFncs
from beamformer import gseidel1  # The benchmark (created with scipy.weave)
#from beamformer_withoutMP import r_beamfull_inverse_OhneMP  # also created with scipy.weave, but WITHOUT using multiple cores via OpenMP

@nb.njit(nb.float32[:](nb.float32[:,:], nb.float32[:], nb.float32[:], nb.int64), cache=True)
def njit_pure(A, dirtyMap, damasSolution, nIterations):
    nGridPoints = len(dirtyMap)
    for cntIter in xrange(nIterations):
        for cntGrid in xrange(nGridPoints):
            solHelp = np.float32(0)
            for cntGridHelp in xrange(cntGrid):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            for cntGridHelp in xrange(cntGrid + 1, nGridPoints):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            solHelp = dirtyMap[cntGrid] - solHelp
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0
    return damasSolution

@nb.njit(nb.float32[:](nb.float32[:,:], nb.float32[:], nb.float32[:], nb.int64), parallel=True)
def njit_parallel(A, dirtyMap, damasSolution, nIterations):
    nGridPoints = len(dirtyMap)
    for cntIter in xrange(nIterations):
        for cntGrid in xrange(nGridPoints):
            solHelp = np.float32(0)
            for cntGridHelp in xrange(cntGrid):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            for cntGridHelp in xrange(cntGrid + 1, nGridPoints):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            solHelp = dirtyMap[cntGrid] - solHelp
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0
    return damasSolution

@nb.guvectorize([(nb.float32[:,:], nb.float32[:], nb.int64, nb.float32[:])], '(g,g),(g),()->(g)', cache=True)
def guvectorizeOverFreqs(A, dirtyMap, nIterations, damasSolution):
    nGridPoints = len(dirtyMap)
    for cntIter in xrange(nIterations):
        for cntGrid in xrange(nGridPoints):
            solHelp = np.float32(0)
            for cntGridHelp in xrange(cntGrid):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            for cntGridHelp in xrange(cntGrid + 1, nGridPoints):
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            solHelp = dirtyMap[cntGrid] - solHelp
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0


def njit_multiThreading(A, dirtyMap, damasSolution, nIterations):
    nGridPoints = len(dirtyMap)
    for cntIter in xrange(nIterations):
        for cntGrid in xrange(nGridPoints):
            sumHelp = np.zeros((2), np.float32)
            threadLowerSum = threading.Thread(target=njit_coreSum, args=(A[cntGrid, :], damasSolution, 0, cntGrid, sumHelp, 0))
            threadUpperSum = threading.Thread(target=njit_coreSum, args=(A[cntGrid, :], damasSolution, cntGrid+1, nGridPoints, sumHelp, 1))
            threadLowerSum.start()
            threadUpperSum.start()
            threadLowerSum.join()
            threadUpperSum.join()
            solHelp = np.float32(dirtyMap[cntGrid] - (sumHelp[0] + sumHelp[1]))
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0
    return damasSolution
@nb.njit(nb.void(nb.float32[:], nb.float32[:], nb.int64, nb.int64, nb.float32[:], nb.int64), cache=True, nogil=True)
def njit_coreSum(A, damasSolution, start, stop, result, indRes):
    for cntGridHelp in xrange(start, stop):
        result[indRes] += A[cntGridHelp] * damasSolution[cntGridHelp]

#%%  MAIN
listOfMics = [0] # not really needed
listGridPoints = [500, 1000, 2000, 5000, 10000]  # Standard value: 12000  # The number of gridpoints doesn't seeme to have to great of an influence
nTrials = 10
nIterations = 10
listOfNFreqs = [20]

funcsToTrial = [njit_pure, njit_parallel, guvectorizeOverFreqs, njit_multiThreading, gseidel1]
for nMics in listOfMics:
    for nGridPoints in listGridPoints:
        for nFreqs in listOfNFreqs:
            # Init
            print(10*'-' + 'New Test configuration: nMics=%s, nGridPoints=%s, nFreqs=%s' %(nMics, nGridPoints, nFreqs) + 10*'-')
            print(10*'-' + 'Creation of inputInputs' + 10*'-')
            
            A = np.float32(np.random.rand(1, nGridPoints, nGridPoints))  #A = np.float32(np.ones((1, nGridPoints, nGridPoints)))
#            for cntFreqs in range(nFreqs):
#                A[cntFreqs, :, :] += A[cntFreqs, :, :].T.conj()
            dirtyMap = np.float32(np.random.rand(1, nGridPoints))  #dirtyMap = np.float32(np.ones((1, nGridPoints)))
            damasSolution = np.zeros((nGridPoints), np.float32)
            
            # create nFreqs times the same matrix for comparing reasons
            A = np.tile(A, [nFreqs, 1, 1])
            dirtyMap = np.tile(dirtyMap, [nFreqs, 1])
            
            nameOfFuncsToTrial = map(lambda x: x.__name__, funcsToTrial)
            nameOfFuncsForError = [funcName for funcName in nameOfFuncsToTrial if funcName != 'gseidel1']
            maxRelativeDeviation = np.zeros((len(funcsToTrial), nTrials))
            maxAbsoluteDeviation = np.zeros((len(funcsToTrial), nTrials))
            timeConsumption = [[] for _ in range(len(funcsToTrial))]
            indOfBaselineFnc = nameOfFuncsToTrial.index('gseidel1')
            
            print(10*'-' + 'Onetime calculation of error reference' + 10*'-')
            gseidel1(A[0,:,:], dirtyMap[0,:], damasSolution, nIterations)
            resultReference = damasSolution  # For relative/absolute error
            gc.collect()
            
            # Testing
            print(10*'-' + 'Testing of functions' + 10*'-')
            cntFunc = 0
            for func in funcsToTrial:
                print(func.__name__)
                for cntTrials in xrange(nTrials):
                    damasSolution = np.zeros((1, nGridPoints), np.float32)
                    damasSolution = np.tile(damasSolution, [nFreqs, 1])
                    if func.__name__ == 'guvectorizeOverFreqs':
                        t0 = tm.time()
                        func(A, dirtyMap, nIterations, damasSolution)
                        t1 = tm.time()
                        result = damasSolution[0, :]
                    elif func.__name__ == 'gseidel1':
                        t0 = tm.time()
                        for cntFreqsHelp in xrange(nFreqs):
                            func(A[cntFreqsHelp,:,:], dirtyMap[cntFreqsHelp,:], damasSolution[cntFreqsHelp,:], nIterations)
                        t1 = tm.time()
                        result = damasSolution[0, :]
                    else:
                        t0 = tm.time()
                        for cntFreqsHelp in xrange(nFreqs):
                            output = func(A[cntFreqsHelp,:,:], dirtyMap[cntFreqsHelp,:], damasSolution[cntFreqsHelp,:], nIterations)
                        t1 = tm.time()
                        result = output
                    timeConsumption[cntFunc].append(t1 - t0)
                    relativeDiffBetweenNewCodeAndRef = (result - resultReference) / (result + resultReference) * 2  # error in relation to the resulting value
                    maxRelativeDeviation[cntFunc, cntTrials] = np.amax(abs(relativeDiffBetweenNewCodeAndRef), axis=0)  # relative error in inf-norm
                    maxAbsoluteDeviation[cntFunc, cntTrials] = np.amax(abs(result - resultReference), axis=0)  # absolute error in inf-norm
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
            stringSaveName = 'Sicherung_DurchgelaufeneTests/damasSolver/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/AllImportantMethods/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/EinflussGridpoints/AMDFX6100/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/JitPrange/' + stringParameters
#            stringSaveName = 'Sicherung_DurchgelaufeneTests/Beamformer/Multithreading_02Threads/' + stringParameters

            shFncs.savingTimeConsumption(stringSaveName, saveTupel)  # saving as "stringSaveName.sav"
            
#            shFncs.plottingOfOvernightTestcasesBeamformer(stringSaveName + '.sav')  # plot of the current test-config
