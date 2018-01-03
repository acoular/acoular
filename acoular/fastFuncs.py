#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------
"""
This file contains all the functionalities which are very expansive, regarding
computational costs. All functionalities are optimized via NUMBA.
"""
import numpy as np
import numba as nb
from six.moves import xrange  # solves the xrange/range issue for python2/3: in py3 'xrange' is now treated as 'range' and in py2 nothing changes

cachedOption = True  # if True: saves the numba func as compiled func in sub directory
parallelOption = 'parallel'  # if numba.guvectorize is used: 'CPU' for single threading; 'parallel' for multithreading; 'cuda' for calculating on GPU


# Formerly known as 'faverage'
@nb.njit(nb.complex128[:,:,:](nb.complex128[:,:,:], nb.complex128[:,:]), cache=cachedOption)
def calcCSM(csm, SpecAllMics):
    """ Adds a given spectrum to the Cross-Spectral-Matrix (CSM).
    Here only the upper triangular matrix of the CSM is calculated. After
    averaging over the various ensembles, the whole CSM is created via complex 
    conjugation transposing. This happens outside 
    (in :class:`PowerSpectra<acoular.spectra.PowerSpectra>`). 
    This method was called 'faverage' in acoular versions <= 16.5.
    
    Parameters
    ----------
    csm : complex128[nFreqs, nMics, nMics] 
        The cross spectral matrix which gets updated with the spectrum of the ensemble.
    SpecAllMics : complex128[nFreqs, nMics] 
        Spectrum of the added ensemble at all Mics.
    
    Returns
    -------
    None : as the input csm gets overwritten.
    """
#==============================================================================
#     It showed, that parallelizing brings no benefit when calling calcCSM once per 
#     ensemble (as its done at the moment). BUT it could be whorth, taking a closer 
#     look to parallelization, when averaging over all ensembles inside this numba 
#     optimized function. See "vglOptimierungFAverage.py" for some information on 
#     the various implementations and their limitations.
#==============================================================================
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in range(nFreqs):
        for cntColumn in range(nMics):
            temp = SpecAllMics[cntFreq, cntColumn].conjugate()
            for cntRow in range(cntColumn + 1):  # calculate upper triangular matrix (of every frequency-slice) only
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllMics[cntFreq, cntRow]
    return csm

    
def beamformerFreq(boolIsEigValProb, steerVecType, boolRemovedDiagOfCSM, normFactor, inputTuple):
    """ Conventional beamformer in frequency domain. Use either a predefined
    steering vector formulation (see Sarradj 2012) or pass it your own
    steering vector.

    Parameters
    ----------
    boolIsEigValProb : bool 
        Should the beamformer use spectral decomposition of the csm matrix?
    steerVecType : (one of the following options: 1, 2, 3, 4, 'custom')
        Either build the steering vector via the predefined formulations
        I - IV (see :ref:`Sarradj, 2012<Sarradj2012>`) or pass it directly.
    boolRemovedDiagOfCSM : bool
        Should the diagonal of the csm be removed?
    normFactor : float
        In here both the signalenergy loss factor (due to removal of the csm diagonal) as well as
        beamforming algorithm (functional, capon, ...) dependent normalization factors are handled.
    inputTuple : dependent of the inputs above. There are 4 combinations:
        boolIsEigValProb = False & steerVecType != 'custom' : 
            inputTuple = (distGridToArrayCenter, distGridToAllMics, waveNumber, csm)
        boolIsEigValProb = False & steerVecType = 'custom' : 
            inputTuple = (steeringVector, csm)
        boolIsEigValProb = True  & steerVecType != 'custom' :
            inputTuple = (distGridToArrayCenter, distGridToAllMics, waveNumber, eigValues, eigVectors)
        boolIsEigValProb = True  & steerVecType = 'custom' :
            inputTuple = (steeringVector, eigValues, eigVectors)

        Data types : In all 4 above cases the data types of inputTuple are
            distGridToArrayCenter : float64[nGridpoints]
                Distance of all gridpoints to the center of sensor array
            distGridToAllMics : float64[nGridpoints, nMics]
                Distance of all gridpoints to all sensors of array
            waveNumber : complex128[nFreqs]
                The wave number should be stored in the imag-part
            csm : complex128[nFreqs, nMics, nMics]
                The cross spectral matrix
            steeringVector : complex128[nFreqs, nGridPoints, nMics]
                The steering vector of each gridpoint and frequency
            eigValues : float64[nFreqs, nEV]
                nEV is the number of eigenvalues which should be taken into account. 
                All passed eigenvalues will be evaluated.
            eigVectors : complex128[nFreqs, nMics, nEV]
                Eigen vectors corresponding to eigValues. All passed eigenvector slices will be evaluated.
    
    Returns
    -------
    Autopower spectrum beamforming map : [nFreqs, nGridPoints]
    
    Some Notes on the optimization of all subroutines
    -------------------------------------------------
    Reducing beamforming equation:
        Let the csm be C and the steering vector be h, than, using Linear Albegra, the conventional beamformer can be written as 
        
        .. math:: B = h^H \\cdot C \\cdot h,
        with ^H meaning the complex conjugated transpose.
        When using that C is a hermitian matrix one can reduce the equation to
        
        .. math:: B = h^H \\cdot C_D \\cdot h + 2 \\cdot Real(h^H \\cdot C_U \\cdot h),
        where C_D and C_U are the diagonal part and upper part of C respectively.
    Steering vector:
        Theoretically the steering vector always includes the term "exp(distMicsGrid - distArrayCenterGrid)", 
        but as the steering vector gets multplied with its complex conjugation in all beamformer routines, 
        the constant "distArrayCenterGrid" cancels out --> In order to save operations, it is not implemented.
    Spectral decomposition of the CSM:
        In Linear Algebra the spectral decomposition of the CSM matrix would be:
        
        .. math:: CSM = \\sum_{i=1}^{nEigenvalues} \\lambda_i (v_i \\cdot v_i^H) ,
        where lambda_i is the i-th eigenvalue and 
        v_i is the eigenvector[nEigVal,1] belonging to lambda_i and ^H denotes the complex conjug transpose. 
        Using this, one must not build the whole CSM (which would be time consuming), but can drag the 
        steering vector into the sum of the spectral decomp. This saves a lot of operations.
    Squares:
        Seemingly "a * a" is slightly faster than "a**2" in numba
    Square of abs():
        Even though "a.real**2 + a.imag**2" would have fewer operations, modern processors seem to be optimized 
        for "a * a.conj" and are slightly faster the latter way. Both Versions are much faster than "abs(a)**2".
    Using Cascading Sums:
        When using the Spectral-Decomposition-Beamformer one could use numpys cascading sums for the scalar product 
        "eigenVec.conj * steeringVector". BUT (at the moment) this only brings benefits in comp-time for a very 
        small range of nMics (approx 250) --> Therefor it is not implemented here.
    """
    # get the beamformer type (key-tuple = (isEigValProblem, formulationOfSteeringVector, RemovalOfCSMDiag))
    beamformerDict = {(False, 1, False) : _freqBeamformer_Formulation1AkaClassic_FullCSM,
                      (False, 1, True) : _freqBeamformer_Formulation1AkaClassic_CsmRemovedDiag,
                      (False, 2, False) : _freqBeamformer_Formulation2AkaInverse_FullCSM,
                      (False, 2, True) : _freqBeamformer_Formulation2AkaInverse_CsmRemovedDiag,
                      (False, 3, False) : _freqBeamformer_Formulation3AkaTrueLevel_FullCSM,
                      (False, 3, True) : _freqBeamformer_Formulation3AkaTrueLevel_CsmRemovedDiag,
                      (False, 4, False) : _freqBeamformer_Formulation4AkaTrueLocation_FullCSM,
                      (False, 4, True) : _freqBeamformer_Formulation4AkaTrueLocation_CsmRemovedDiag,
                      (False, 'custom', False) : _freqBeamformer_SpecificSteerVec_FullCSM,
                      (False, 'custom', True) : _freqBeamformer_SpecificSteerVec_CsmRemovedDiag,
                      (True, 1, False) : _freqBeamformer_EigValProb_Formulation1AkaClassic_FullCSM,
                      (True, 1, True) : _freqBeamformer_EigValProb_Formulation1AkaClassic_CsmRemovedDiag,
                      (True, 2, False) : _freqBeamformer_EigValProb_Formulation2AkaInverse_FullCSM,
                      (True, 2, True) : _freqBeamformer_EigValProb_Formulation2AkaInverse_CsmRemovedDiag,
                      (True, 3, False) : _freqBeamformer_EigValProb_Formulation3AkaTrueLevel_FullCSM,
                      (True, 3, True) : _freqBeamformer_EigValProb_Formulation3AkaTrueLevel_CsmRemovedDiag,
                      (True, 4, False) : _freqBeamformer_EigValProb_Formulation4AkaTrueLocation_FullCSM,
                      (True, 4, True) : _freqBeamformer_EigValProb_Formulation4AkaTrueLocation_CsmRemovedDiag,
                      (True, 'custom', False) : _freqBeamformer_EigValProb_SpecificSteerVec_FullCSM,
                      (True, 'custom', True) : _freqBeamformer_EigValProb_SpecificSteerVec_CsmRemovedDiag,}
    coreFunc = beamformerDict[(boolIsEigValProb, steerVecType, boolRemovedDiagOfCSM)]

    # prepare Input
    if steerVecType == 'custom':  # beamformer with custom steering vector
        steerVec = inputTuple[0]
        nFreqs, nGridPoints = steerVec.shape[0], steerVec.shape[1]
        if boolIsEigValProb:
            eigVal, eigVec = inputTuple[1], inputTuple[2]
        else:
            csm = inputTuple[1]
    else:  # predefined beamformers (Formulation I - IV)
        distGridToArrayCenter, distGridToAllMics, waveNumber = inputTuple[0], inputTuple[1], inputTuple[2]
        nFreqs, nGridPoints = waveNumber.shape[0], distGridToAllMics.shape[0]
        if boolIsEigValProb:
            eigVal, eigVec = inputTuple[3], inputTuple[4]
        else:
            csm = inputTuple[3]
    
    # beamformer routine: parallelized over Gridpoints
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros(nGridPoints, np.float64)
        if steerVecType == 'custom':  # beamformer with custom steering vector
            if boolIsEigValProb:
                coreFunc(eigVal[cntFreqs, :], eigVec[cntFreqs, :, :], steerVec[cntFreqs, :, :], normFactor, result)
            else:
                coreFunc(csm[cntFreqs, :, :], steerVec[cntFreqs, :, :], normFactor, result)
        else:  # predefined beamformers (Formulation I - IV)
            if boolIsEigValProb:
                coreFunc(eigVal[cntFreqs, :], eigVec[cntFreqs, :, :], distGridToArrayCenter, distGridToAllMics, waveNumber[cntFreqs].imag, normFactor, result)
            else:
                coreFunc(csm[cntFreqs, :, :], distGridToArrayCenter, distGridToAllMics, waveNumber[cntFreqs].imag, normFactor, result)
        beamformOutput[cntFreqs, :] = result
    return beamformOutput


#%% beamformers - steer * CSM * steer
@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation1AkaClassic_FullCSM(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg))

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq)
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    normalizeFactor = nMics  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation1AkaClassic_CsmRemovedDiag(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg))

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    normalizeFactor = nMics  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation2AkaInverse_FullCSM(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    normalizeFactor = nMics * distGridToArrayCenter[0]  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation2AkaInverse_CsmRemovedDiag(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    normalizeFactor = nMics * distGridToArrayCenter[0]  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation3AkaTrueLevel_FullCSM(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    normalizeFactor = distGridToArrayCenter[0] * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation3AkaTrueLevel_CsmRemovedDiag(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    normalizeFactor = distGridToArrayCenter[0] * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation4AkaTrueLocation_FullCSM(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    normalizeFactor = nMics * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProd / normalizeFactor * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(m,m),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_Formulation4AkaTrueLocation_CsmRemovedDiag(csm, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    normalizeFactor = nMics * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProd / normalizeFactor * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:])], '(m,m),(m),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_SpecificSteerVec_FullCSM(csm, steerVec, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    result[0] = scalarProd * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:])], '(m,m),(m),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_SpecificSteerVec_CsmRemovedDiag(csm, steerVec, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    for cntMics in xrange(nMics):
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in xrange(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    result[0] = scalarProd * signalLossNormalization[0]


#%% beamformers - Eigenvalue Problem

@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation1AkaClassic_FullCSM(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg))

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    normalizeFactor = nMics  # specific normalization of steering vector formulation
    result[0] = scalarProdFullCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation1AkaClassic_CsmRemovedDiag(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg))

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdReducedCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        scalarProdDiagCSMperEigVal = 0.0
        for cntMics in range(nMics):
            temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]  # Dont call it 'expArg' like in steer-loop, because expArg is now a float (no double) which would cause errors of approx 1e-8
            scalarProdFullCSMperEigVal += temp1
            scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
        scalarProdReducedCSM += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
    normalizeFactor = nMics  # specific normalization of steering vector formulation
    result[0] = scalarProdReducedCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation2AkaInverse_FullCSM(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    normalizeFactor = nMics * distGridToArrayCenter[0]  # specific normalization of steering vector formulation
    result[0] = scalarProdFullCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation2AkaInverse_CsmRemovedDiag(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdReducedCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        scalarProdDiagCSMperEigVal = 0.0
        for cntMics in range(nMics):
            temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]  # Dont call it 'expArg' like in steer-loop, because expArg is now a float (no double) which would cause errors of approx 1e-8
            scalarProdFullCSMperEigVal += temp1
            scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
        scalarProdReducedCSM += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
    normalizeFactor = nMics * distGridToArrayCenter[0]  # specific normalization of steering vector formulation
    result[0] = scalarProdReducedCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation3AkaTrueLevel_FullCSM(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    normalizeFactor = distGridToArrayCenter[0] * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProdFullCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation3AkaTrueLevel_CsmRemovedDiag(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdReducedCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        scalarProdDiagCSMperEigVal = 0.0
        for cntMics in range(nMics):
            temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]  # Dont call it 'expArg' like in steer-loop, because expArg is now a float (no double) which would cause errors of approx 1e-8
            scalarProdFullCSMperEigVal += temp1
            scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
        scalarProdReducedCSM += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
    normalizeFactor = distGridToArrayCenter[0] * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProdReducedCSM / (normalizeFactor * normalizeFactor) * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation4AkaTrueLocation_FullCSM(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    normalizeFactor = nMics * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProdFullCSM / normalizeFactor * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
              '(e),(m,e),(),(m),(),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_Formulation4AkaTrueLocation_CsmRemovedDiag(eigVal, eigVec, distGridToArrayCenter, distGridToAllMics, waveNumber, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = distGridToAllMics.shape[0]
    steerVec = np.zeros((nMics), np.complex128)

    # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
    helpNormalize = 0.0
    for cntMics in xrange(nMics):
        helpNormalize += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])  
        expArg = np.float32(waveNumber[0] * distGridToAllMics[cntMics])
        steerVec[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) / distGridToAllMics[cntMics]  # r_{t,i}-normalization is handled here

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdReducedCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        scalarProdDiagCSMperEigVal = 0.0
        for cntMics in range(nMics):
            temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]  # Dont call it 'expArg' like in steer-loop, because expArg is now a float (no double) which would cause errors of approx 1e-8
            scalarProdFullCSMperEigVal += temp1
            scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
        scalarProdReducedCSM += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
    normalizeFactor = nMics * helpNormalize  # specific normalization of steering vector formulation
    result[0] = scalarProdReducedCSM / normalizeFactor * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:])],
                 '(e),(m,e),(m),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_SpecificSteerVec_FullCSM(eigVal, eigVec, steerVec, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = eigVec.shape[0]

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    result[0] = scalarProdFullCSM * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:])],
                 '(e),(m,e),(m),()->()', nopython=True, target=parallelOption, cache=cachedOption)
def _freqBeamformer_EigValProb_SpecificSteerVec_CsmRemovedDiag(eigVal, eigVec, steerVec, signalLossNormalization, result):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = eigVec.shape[0]

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdReducedCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        scalarProdDiagCSMperEigVal = 0.0
        for cntMics in range(nMics):
            temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
            scalarProdFullCSMperEigVal += temp1
            scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
        scalarProdReducedCSM += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
    result[0] = scalarProdReducedCSM * signalLossNormalization[0]

#%% Point - Spread - Function
def calcPointSpreadFunction(steerVecType, inputTuple):
    """ Calculates the Point-Spread-Functions. Use either a predefined steering vector 
    formulation (see :ref:`Sarradj, 2012<Sarradj2012>`) or pass it your own steering vector.

    Parameters
    ----------
    steerVecType : (one of the following options: 1, 2, 3, 4, 'custom')
        Either build the steering vector via the predefined formulations
        I - IV (see :ref:`Sarradj, 2012<Sarradj2012>`) or pass it directly. (not implemented yet).
    inputTuple: dependent of the inputs above. If
        steerVecType != 'custom' :
            inputTuple =(distGridToArrayCenter, distGridToAllMics, waveNumber, indSource)
        steerVecType = 'custom' :
            NOT IMPLEMENTED YET!!!!!!    #inputTuple =(steeringVector, transfer, indSource)
    
        Data types : In both cases the data types of inputTuple are
            distGridToArrayCenter : float64[nGridpoints]
                Distance of all gridpoints to the center of sensor array
            distGridToAllMics : float64[nGridpoints, nMics]
                Distance of all gridpoints to all sensors of array
            waveNumber : complex128[nFreqs]
                The wave number should be stored in the imag-part
            indSource : a LIST of int (e.g. indSource=[5] is fine; indSource=5 doesn't work):
                specifies which gridpoints should be assumed to be sources 
                --> a seperate psf will be calculated for each source

    Returns
    -------
    Autopower spectrum PSF map : [nFreqs, nGridPoints, nSources]
    
    Some Notes on the optimization of all subroutines
    -------------------------------------------------
    Reducing beamforming equation:
        Let the steering vector be h, than, using Linear Albegra, the PSF of a SourcePoint S would be
        
        .. math:: B = h^H \\cdot (a_S \\cdot a_S^H) \\cdot h,
        with ^H meaning the complex conjugated transpose and a_s the transfer function from source to gridpoint.
        The (...)-part equals the CSM that the source would produce via the chosen steering vec formulation. 
        Using (for example) tensor calculus, one can reduce the equation to:
        
        .. math:: B = \\left| h^H \\cdot a_S \\right| ^ 2.
    Steering vector:
        Theoretically the steering vector always includes the term "exp(distMicsGrid - distArrayCenterGrid)", but as the steering vector gets multplied with its complex conjugation in 
        all beamformer routines, the constant "distArrayCenterGrid" cancels out --> In order to save operations, it is not implemented.
    Squares:
        Seemingly "a * a" is slightly faster than "a**2" in numba
    Square of abs():
        Even though "a.real**2 + a.imag**^2" would have fewer operations, modern processors seem to be optimized for "a * a.conj" and are slightly faster the latter way.
        Both Versions are much faster than "abs(a)**2".
    """
    # get the steering vector formulation
    psfDict = {1 : _psf_Formulation1AkaClassic,
               2 : _psf_Formulation2AkaInverse,
               3 : _psf_Formulation3AkaTrueLevel,
               4 : _psf_Formulation4AkaTrueLocation}#,
#               'custom' : _psf_SpecificSteerVec}
    coreFunc = psfDict[steerVecType]

    # prepare input
    if steerVecType == 'custom':  # PSF with custom steering vector
        raise ValueError('custom Steering vectors are not implemented yet.')
#        steerVec, transFunc, indSource = inputTuple[0], inputTuple[1], inputTuple[2]
#        nFreqs, nGridPoints = steerVec.shape[0], steerVec.shape[1]
    else:  # predefined steering vectors (Formulation I - IV)
        distGridToArrayCenter, distGridToAllMics, waveNumber, indSource = inputTuple[0], inputTuple[1], inputTuple[2], inputTuple[3]
        nFreqs, nGridPoints = waveNumber.shape[0], distGridToAllMics.shape[0]
    nSources = len(indSource)

    # psf routine: parallelized over Gridpoints
    psfOutput = np.zeros((nFreqs, nGridPoints, nSources), np.float64)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros((nGridPoints, nSources), np.float64)
        if steerVecType == 'custom':
            raise ValueError('custom Steering vectors are not implemented yet.')
#            coreFunc(steerVec[cntFreqs, :, :], transFunc[cntFreqs, indSource, :], result)
        else:  # predefined steering vector (Formulation I - IV)
            coreFunc(distGridToArrayCenter, distGridToAllMics, distGridToArrayCenter[indSource], distGridToAllMics[indSource, :], waveNumber[cntFreqs].imag, result)
        psfOutput[cntFreqs, :, :] = result
    return psfOutput


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:])], '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption)
def _psf_Formulation1AkaClassic(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        for cntMics in xrange(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics]
        normalizeFactor = distSourcesToArrayCenter[cntSources] / nMics
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:])], '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption)
def _psf_Formulation2AkaInverse(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        for cntMics in xrange(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] * distGridToAllMics[cntMics]
        normalizeFactor = distSourcesToArrayCenter[cntSources] / distGridToArrayCenter[0] / nMics
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real  
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)  


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:])], '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption)
def _psf_Formulation3AkaTrueLevel(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        helpNormalizeGrid = 0.0
        for cntMics in xrange(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] / distGridToAllMics[cntMics]
            helpNormalizeGrid += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])
        normalizeFactor = distSourcesToArrayCenter[cntSources] / distGridToArrayCenter[0] / helpNormalizeGrid
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:])], '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption)
def _psf_Formulation4AkaTrueLocation(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        helpNormalizeGrid = 0.0
        for cntMics in xrange(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] / distGridToAllMics[cntMics]
            helpNormalizeGrid += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])
        normalizeFactor = distSourcesToArrayCenter[cntSources]
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor) / nMics / helpNormalizeGrid / helpNormalizeGrid  # ERROR: one helpNormalizeGrid too much! needs to be corrected after py3 migration

# NEEDS TO BE OVERLOOKED!!
#@nb.guvectorize([(nb.complex128[:], nb.complex128[:,:], nb.float64[:])], '(m),(s,m)->(s)', nopython=True, target=parallelOption, cache=cachedOption)
#def _psf_SpecificSteerVec(steerVec, steerVecSources, result):
#    nMics = len(steerVec)
#    for cntSources in range(steerVecSources.shape[0]):
#        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
#        scalarProd = 0.0 + 0.0j
#        for cntMics in xrange(nMics):
#            scalarProd += steerVec[cntMics].conjugate() * steerVecSources[cntSources, cntMics]
#        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real  
#        result[cntSources] = scalarProdAbsSquared


#%% Damas - Gauss Seidel
# Formerly known as 'gseidel'
@nb.guvectorize([(nb.float32[:,:], nb.float32[:], nb.int64[:], nb.float64[:], nb.float32[:]), 
                 (nb.float64[:,:], nb.float64[:], nb.int64[:], nb.float64[:], nb.float64[:])], '(g,g),(g),(),()->(g)', nopython=True, target=parallelOption, cache=cachedOption)
def damasSolverGaussSeidel(A, dirtyMap, nIterations, relax, damasSolution):
    """ Solves the DAMAS inverse problem via modified gauss seidel.
    This is the original formulation from :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`.
    
    Parameters
    ----------
    A : float32[nFreqs, nGridpoints, nGridpoints] (or float64[...])
        The PSF build matrix (see :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`)
    dirtyMap : float32[nFreqs, nGridpoints] (or float64[...])
        The conventional beamformer map
    nIterations : int64[scalar] 
        number of Iterations the damas solver has to go through
    relax : int64[scalar] 
        relaxation parameter (=1.0 in :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`)
    damasSolution : float32[nFreqs, nGridpoints] (or float64[...]) 
        starting solution
    
    Returns
    -------
    None : as damasSolution is overwritten with end result of the damas iterative solver.
    """
    nGridPoints = len(dirtyMap)
    for cntIter in xrange(nIterations[0]):
        for cntGrid in xrange(nGridPoints):
            solHelp = np.float32(0)
            for cntGridHelp in xrange(cntGrid):  # lower sum
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            for cntGridHelp in xrange(cntGrid + 1, nGridPoints):  # upper sum
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            solHelp = (1 - relax[0]) * damasSolution[cntGrid] + relax[0] * (dirtyMap[cntGrid] - solHelp)
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0


#%% Transfer - Function
def transfer(distGridToArrayCenter, distGridToAllMics, waveNumber):
    """ Calculates the transfer functions between the various mics and gridpoints.
    
    Parameters
    ----------
    distGridToArrayCenter : float64[nGridpoints]
        Distance of all gridpoints to the center of sensor array
    distGridToAllMics : float64[nGridpoints, nMics]
        Distance of all gridpoints to all sensors of array
    waveNumber : complex128[nFreqs]
        The wave number should be stored in the imag-part

    Returns
    -------
    The Transferfunctions in format complex128[nFreqs, nGridPoints, nMics].
    """
    nFreqs, nGridPoints, nMics = waveNumber.shape[0], distGridToAllMics.shape[0], distGridToAllMics.shape[1]
    # transfer routine: parallelized over Gridpoints
    transferOutput = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
    for cntFreqs in xrange(nFreqs):
        result = np.zeros((nGridPoints, nMics), np.complex128)
        _transferCoreFunc(distGridToArrayCenter, distGridToAllMics, waveNumber[cntFreqs].imag, result)
        transferOutput[cntFreqs, :, :] = result
    return transferOutput

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.complex128[:])], '(),(m),()->(m)', nopython=True, target=parallelOption, cache=cachedOption)
def _transferCoreFunc(distGridToArrayCenter, distGridToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntMics in xrange(nMics):
        expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distGridToArrayCenter[0]))  # FLOAT32 ODER FLOAT64?
        result[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToArrayCenter[0] / distGridToAllMics[cntMics]