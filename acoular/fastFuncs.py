#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""
This file contains all the functionalities which are very expansive, regarding
computational costs. All functionalities are optimized via NUMBA.
"""
import numpy as np
import numba as nb

cachedOption = True  # if True: saves the numba func as compiled func in sub directory
parallelOption = 'parallel'  # if numba.guvectorize is used: 'CPU' for single threading; 'parallel' for multithreading; 'cuda' for calculating on GPU
fastOption = True # fastmath options 


# Formerly known as 'faverage'
@nb.njit([nb.complex128[:,:,::1](nb.complex128[:,:,::1], nb.complex128[:,::1]), 
          nb.complex64[:,:,::1](nb.complex64[:,:,::1], nb.complex64[:,::1])], cache=cachedOption, parallel=True, fastmath=fastOption)
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
    nFreqs = csm.shape[0]
    nMics = csm.shape[1]
    for cntFreq in nb.prange(nFreqs):
        for cntColumn in range(nMics):
            temp = SpecAllMics[cntFreq, cntColumn].conjugate()
            for cntRow in range(cntColumn + 1):  # calculate upper triangular matrix (of every frequency-slice) only
                csm[cntFreq, cntRow, cntColumn] += temp * SpecAllMics[cntFreq, cntRow]
    return csm

    
def beamformerFreq(steerVecType, boolRemovedDiagOfCSM, normFactor, inputTupleSteer, inputTupleCsm):
    """ Conventional beamformer in frequency domain. Use either a predefined
    steering vector formulation (see Sarradj 2012) or pass your own
    steering vector.

    Parameters
    ----------
    steerVecType : (one of the following strings: 'classic' (I), 'inverse' (II), 'true level' (III), 'true location' (IV), 'custom')
        Either build the steering vector via the predefined formulations
        I - IV (see :ref:`Sarradj, 2012<Sarradj2012>`) or pass it directly.
    boolRemovedDiagOfCSM : bool
        Should the diagonal of the csm be removed?
    normFactor : float
        In here both the signalenergy loss factor (due to removal of the csm diagonal) as well as
        beamforming algorithm (music, capon, ...) dependent normalization factors are handled.
    inputTupleSteer : contains the information needed to create the steering vector. Is dependent of steerVecType. There are 2 cases:
        steerVecType != 'custom' :
            inputTupleSteer = (distGridToArrayCenter, distGridToAllMics, waveNumber)    , with
                distGridToArrayCenter : float64[nGridpoints]
                    Distance of all gridpoints to the center of sensor array
                distGridToAllMics : float64[nGridpoints, nMics]
                    Distance of all gridpoints to all sensors of array
                waveNumber : float64
                    The wave number
        steerVecType == 'custom' :
            inputTupleSteer = steeringVector    , with
                steeringVector : complex128[nGridPoints, nMics]
                    The steering vector of each gridpoint for the same frequency as the CSM
    inputTupleCsm : contains the data of measurement as a tuple. There are 2 cases:
        perform standard CSM-beamformer:
            inputTupleCsm = csm
                csm : complex128[ nMics, nMics]
                    The cross spectral matrix for one frequency
        perform beamformer on eigenvalue decomposition of csm:
            inputTupleCsm = (eigValues, eigVectors)    , with
                eigValues : float64[nEV]
                    nEV is the number of eigenvalues which should be taken into account. 
                    All passed eigenvalues will be evaluated.
                eigVectors : complex128[nMics, nEV]
                    Eigen vectors corresponding to eigValues. All passed eigenvector slices will be evaluated.

    Returns
    -------
    *Autopower spectrum beamforming map [nGridPoints] 
         
    *steer normalization factor [nGridPoints]... contains the values the autopower needs to be multiplied with, in order to 
    fullfill 'steer^H * steer = 1' as needed for functional beamforming. 
    
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
    boolIsEigValProb = isinstance(inputTupleCsm, tuple)# len(inputTupleCsm) > 1
    # get the beamformer type (key-tuple = (isEigValProblem, formulationOfSteeringVector, RemovalOfCSMDiag))
    beamformerDict = {(False, 'custom', False) : _freqBeamformer_SpecificSteerVec_FullCSM,
                      (False, 'custom', True) : _freqBeamformer_SpecificSteerVec_CsmRemovedDiag,
                      (True, 'custom', False) : _freqBeamformer_EigValProb_SpecificSteerVec_FullCSM,
                      (True, 'custom', True) : _freqBeamformer_EigValProb_SpecificSteerVec_CsmRemovedDiag}
    sth = {'classic':1, 'inverse':2,'true level':3, 'true location':4}
   
    # prepare Input
    if steerVecType == 'custom':  # beamformer with custom steering vector
        steerVec = inputTupleSteer
        nGridPoints = steerVec.shape[0]
    else:  # predefined beamformers (Formulation I - IV)
        distGridToArrayCenter, distGridToAllMics, waveNumber = inputTupleSteer
        if not isinstance(waveNumber, np.ndarray): waveNumber = np.array([waveNumber]) #for backward compatibility
        nGridPoints = distGridToAllMics.shape[0]
    if boolIsEigValProb:
        eigVal, eigVec = inputTupleCsm#[0], inputTupleCsm[1]
    else:
        csm = inputTupleCsm
    
    # beamformer routine: parallelized over Gridpoints
    beamformOutput = np.zeros(nGridPoints, np.float64)
    steerNormalizeOutput = np.zeros_like(beamformOutput)
    result = np.zeros(nGridPoints, np.float64)
    normalHelp = np.zeros_like(result)
    if steerVecType == 'custom':  # beamformer with custom steering vector
        coreFunc = beamformerDict[(boolIsEigValProb, steerVecType, boolRemovedDiagOfCSM)]
        if boolIsEigValProb:
            coreFunc(eigVal, eigVec, steerVec, normFactor, result, normalHelp)
        else:
            coreFunc(csm, steerVec, normFactor, result, normalHelp)
    else:  # predefined beamformers (Formulation I - IV)
        if boolIsEigValProb:
            _freqBeamformer_EigValues(eigVal, np.ascontiguousarray(eigVec), distGridToArrayCenter, distGridToAllMics, waveNumber[0], normFactor, 
                                    boolRemovedDiagOfCSM, sth[steerVecType],
                                    result, normalHelp)
        else:
            _freqBeamformer_FullCSM(csm, distGridToArrayCenter, distGridToAllMics, waveNumber[0], normFactor, 
                                    boolRemovedDiagOfCSM, sth[steerVecType],
                                    result, normalHelp)
    beamformOutput = result
    steerNormalizeOutput = normalHelp 
    return beamformOutput, steerNormalizeOutput 

# fast implementation of full matrix beamformers
@nb.njit(
    [
        (
            nb.complex128[:, ::1],
            nb.float64[::1],
            nb.float64[:, ::1],
            nb.float64,
            nb.float64,
            nb.boolean,
            nb.int64,
            nb.float64[::1],
            nb.float64[::1],
        )
    ],
    cache=cachedOption,
    parallel=True,
    error_model="numpy"
)
def _freqBeamformer_FullCSM(
    csm,
    distGridToArrayCenter,
    distGridToAllMics,
    waveNumber,
    signalLossNormalization,
    r_diag,
    steer_type,
    result,
    normalizeSteer,
):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]
    st2 = (steer_type == 2)
    st34 = (steer_type == 3 or steer_type == 4)
    helpNormalize = 0.0 # just a hint for the compiler
    for gi in nb.prange(distGridToArrayCenter.shape[0]):
        steerVec = np.empty((nMics), np.complex128)
        # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber * distGridToAllMics[gi, cntMics])
            steerVec[cntMics] = np.cos(expArg) - 1j * np.sin(expArg)
        if st2:    
            helpNormalize = 0.0
            for cntMics in range(nMics):
                helpNormalize += distGridToAllMics[gi,cntMics] * distGridToAllMics[gi,cntMics]
                steerVec[cntMics] *= distGridToAllMics[gi,cntMics]  # r_{t,i}-normalization is handled here
        if st34:
            helpNormalize = 0.0
            for cntMics in range(nMics):
                helpNormalize += 1.0 / (distGridToAllMics[gi,cntMics] * distGridToAllMics[gi,cntMics])  
                steerVec[cntMics] /= distGridToAllMics[gi,cntMics]  # r_{t,i}-normalization is handled here

        # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq)
        scalarProd = 0.0
        for cntMics in range(nMics):
            leftVecMatrixProd = 0.0 + 0.0j
            for cntMics2 in range(
                cntMics
            ):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
                leftVecMatrixProd += (
                    csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
                )
            scalarProd += (
                2 * (leftVecMatrixProd * steerVec[cntMics]).real
            )  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        if not r_diag:
            for cntMics in range(nMics):
                scalarProd += (
                    csm[cntMics, cntMics]
                    * steerVec[cntMics].conjugate()
                    * steerVec[cntMics]
                ).real  # include diagonal of csm

        # specific normalzation for different steering vector formulations
        if steer_type == 1:
            normalizeFactor = nMics
            normalizeSteer[gi] = 1.0 / nMics
            result[gi] = (
                scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization
            )
        elif steer_type == 2:
            normalizeFactor = nMics * distGridToArrayCenter[gi]
            normalizeFactorSquared = normalizeFactor * normalizeFactor
            normalizeSteer[gi] = helpNormalize / normalizeFactorSquared
            result[gi] = scalarProd / normalizeFactorSquared * signalLossNormalization
        elif steer_type == 3:
            normalizeFactor = distGridToArrayCenter[gi] * helpNormalize
            normalizeSteer[gi] = 1.0 / (distGridToArrayCenter[gi] * distGridToArrayCenter[gi]) / helpNormalize
            result[gi] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization
        elif steer_type == 4:
            normalizeFactor = nMics * helpNormalize
            normalizeSteer[gi] = 1.0 / nMics
            result[gi] = scalarProd / normalizeFactor * signalLossNormalization

# fast implementation of eigenvalue beamformers
@nb.njit(
    [
        (
            nb.float64[::1],
            nb.complex128[:, ::1],
            nb.float64[::1],
            nb.float64[:, ::1],
            nb.float64,
            nb.float64,
            nb.boolean,
            nb.int64,
            nb.float64[::1],
            nb.float64[::1],
        )
    ],
    cache=cachedOption,
    parallel=True,
    error_model="numpy"
)
def _freqBeamformer_EigValues(
    eigVal,
    eigVec,
    distGridToArrayCenter,
    distGridToAllMics,
    waveNumber,
    signalLossNormalization,
    r_diag,
    steer_type,
    result,
    normalizeSteer,
):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = eigVec.shape[0]
    nEigs = len(eigVal)
    st2 = (steer_type == 2)
    st34 = (steer_type == 3 or steer_type == 4)
    helpNormalize = 0.0 # just a hint for the compiler
    for gi in nb.prange(distGridToArrayCenter.shape[0]):
        steerVec = np.empty((nMics), np.complex128)
        # building steering vector: in order to save some operation -> some normalization steps are applied after mat-vec-multipl.
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber * distGridToAllMics[gi, cntMics])
            steerVec[cntMics] = np.cos(expArg) - 1j * np.sin(expArg)
        if st2:    
            helpNormalize = 0.0
            for cntMics in range(nMics):
                helpNormalize += distGridToAllMics[gi,cntMics] * distGridToAllMics[gi,cntMics]
                steerVec[cntMics] *= distGridToAllMics[gi,cntMics]  # r_{t,i}-normalization is handled here
        if st34:
            helpNormalize = 0.0
            for cntMics in range(nMics):
                helpNormalize += 1.0 / (distGridToAllMics[gi,cntMics] * distGridToAllMics[gi,cntMics])  
                steerVec[cntMics] /= distGridToAllMics[gi,cntMics]  # r_{t,i}-normalization is handled here

        # eigenvalue beamforming    
        scalarProd = 0.0
        if r_diag:
            for cntEigVal in range(len(eigVal)):
                scalarProdFullCSMperEigVal = 0.0 + 0.0j
                scalarProdDiagCSMperEigVal = 0.0
                for cntMics in range(nMics):
                    temp1 = eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]  
                    scalarProdFullCSMperEigVal += temp1
                    scalarProdDiagCSMperEigVal += (temp1 * temp1.conjugate()).real  
                scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real
                scalarProd += (scalarProdFullCSMAbsSquared - scalarProdDiagCSMperEigVal) * eigVal[cntEigVal]
        else:
            for cntEigVal in range(nEigs):
                scalarProdFullCSMperEigVal = 0.0 + 0.0j
                for cntMics in range(nMics):
                    scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
                scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
                scalarProd += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]

        # specific normalzation for different steering vector formulations
        if steer_type == 1:
            normalizeFactor = nMics
            normalizeSteer[gi] = 1.0 / nMics
            result[gi] = (
                scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization
            )
        elif steer_type == 2:
            normalizeFactor = nMics * distGridToArrayCenter[gi]
            normalizeFactorSquared = normalizeFactor * normalizeFactor
            normalizeSteer[gi] = helpNormalize / normalizeFactorSquared
            result[gi] = scalarProd / normalizeFactorSquared * signalLossNormalization
        elif steer_type == 3:
            normalizeFactor = distGridToArrayCenter[gi] * helpNormalize
            normalizeSteer[gi] = 1.0 / (distGridToArrayCenter[gi] * distGridToArrayCenter[gi]) / helpNormalize
            result[gi] = scalarProd / (normalizeFactor * normalizeFactor) * signalLossNormalization
        elif steer_type == 4:
            normalizeFactor = nMics * helpNormalize
            normalizeSteer[gi] = 1.0 / nMics
            result[gi] = scalarProd / normalizeFactor * signalLossNormalization

@nb.guvectorize([(nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:])], 
                '(m,m),(m),()->(),()', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _freqBeamformer_SpecificSteerVec_FullCSM(csm, steerVec, signalLossNormalization, result, normalizeSteer):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    helpNormalize = 0.0
    for cntMics in range(nMics):
        helpNormalize += steerVec[cntMics] * steerVec[cntMics].conjugate()
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in range(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
        scalarProd += (csm[cntMics, cntMics] * steerVec[cntMics].conjugate() * steerVec[cntMics]).real  # include diagonal of csm
    normalizeSteer[0] = helpNormalize.real
    result[0] = scalarProd * signalLossNormalization[0]


@nb.guvectorize([(nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:])], 
                '(m,m),(m),()->(),()', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _freqBeamformer_SpecificSteerVec_CsmRemovedDiag(csm, steerVec, signalLossNormalization, result, normalizeSteer):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = csm.shape[0]

    # performing matrix-vector-multiplication (see bottom of information header of 'beamformerFreq')
    scalarProd = 0.0
    helpNormalize = 0.0
    for cntMics in range(nMics):
        helpNormalize += steerVec[cntMics] * steerVec[cntMics].conjugate()
        leftVecMatrixProd = 0.0 + 0.0j
        for cntMics2 in range(cntMics):  # calculate 'steer^H * CSM' of upper-triangular-part of csm (without diagonal)
            leftVecMatrixProd += csm[cntMics2, cntMics] * steerVec[cntMics2].conjugate()
        scalarProd += 2 * (leftVecMatrixProd * steerVec[cntMics]).real  # use that csm is Hermitian (lower triangular of csm can be reduced to factor '2')
    normalizeSteer[0] = helpNormalize.real
    result[0] = scalarProd * signalLossNormalization[0]
@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                 '(e),(m,e),(m),()->(),()', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _freqBeamformer_EigValProb_SpecificSteerVec_FullCSM(eigVal, eigVec, steerVec, signalLossNormalization, result, normalizeSteer):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = eigVec.shape[0]
    
    # get h^H * h for normalization
    helpNormalize = 0.0
    for cntMics in range(nMics):
        helpNormalize += steerVec[cntMics] * steerVec[cntMics].conjugate()

    # performing matrix-vector-multplication via spectral decomp. (see bottom of information header of 'beamformerFreq')
    scalarProdFullCSM = 0.0
    for cntEigVal in range(len(eigVal)):
        scalarProdFullCSMperEigVal = 0.0 + 0.0j
        for cntMics in range(nMics):
            scalarProdFullCSMperEigVal += eigVec[cntMics, cntEigVal].conjugate() * steerVec[cntMics]
        scalarProdFullCSMAbsSquared = (scalarProdFullCSMperEigVal * scalarProdFullCSMperEigVal.conjugate()).real  
        scalarProdFullCSM += scalarProdFullCSMAbsSquared * eigVal[cntEigVal]
    normalizeSteer[0] = helpNormalize.real
    result[0] = scalarProdFullCSM * signalLossNormalization[0]


@nb.guvectorize([(nb.float64[:], nb.complex128[:,:], nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                 '(e),(m,e),(m),()->(),()', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _freqBeamformer_EigValProb_SpecificSteerVec_CsmRemovedDiag(eigVal, eigVec, steerVec, signalLossNormalization, result, normalizeSteer):
    # see bottom of information header of 'beamformerFreq' for information on which steps are taken, in order to gain speed improvements.
    nMics = eigVec.shape[0]
    
    # get h^H * h for normalization
    helpNormalize = 0.0
    for cntMics in range(nMics):
        helpNormalize += steerVec[cntMics] * steerVec[cntMics].conjugate()

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
    normalizeSteer[0] = helpNormalize.real
    result[0] = scalarProdReducedCSM * signalLossNormalization[0]

#%% Point - Spread - Function
def calcPointSpreadFunction(steerVecType, distGridToArrayCenter, distGridToAllMics, waveNumber, indSource, dtype):
    """ Calculates the Point-Spread-Functions. Use either a predefined steering vector 
    formulation (see :ref:`Sarradj, 2012<Sarradj2012>`) or pass it your own steering vector.

    Parameters
    ----------
    steerVecType : (one of the following strings: 'classic' (I), 'inverse' (II), 'true level' (III), 'true location' (IV))
        One of the predefined formulations I - IV (see :ref:`Sarradj, 2012<Sarradj2012>`).
    distGridToArrayCenter : float64[nGridpoints]
        Distance of all gridpoints to the center of sensor array
    distGridToAllMics : float64[nGridpoints, nMics]
        Distance of all gridpoints to all sensors of array
    waveNumber : float64
        The free field wave number.
    indSource : a LIST of int (e.g. indSource=[5] is fine; indSource=5 doesn't work):
        specifies which gridpoints should be assumed to be sources 
        --> a seperate psf will be calculated for each source
    dtype : either 'float64' or 'float32'
        Determines the precision of the result. For big maps this could be worth downgrading.

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
    psfDict = {'classic' : _psf_Formulation1AkaClassic,
               'inverse' : _psf_Formulation2AkaInverse,
               'true level' : _psf_Formulation3AkaTrueLevel,
               'true location' : _psf_Formulation4AkaTrueLocation}
    coreFunc = psfDict[steerVecType]

    # prepare input
    nGridPoints = distGridToAllMics.shape[0]
    nSources = len(indSource)
    if not isinstance(waveNumber, np.ndarray): waveNumber = np.array([waveNumber])
    
    # psf routine: parallelized over Gridpoints
    psfOutput = np.zeros((nGridPoints, nSources), dtype=dtype)
    coreFunc(distGridToArrayCenter, 
             distGridToAllMics, 
             distGridToArrayCenter[indSource], 
             distGridToAllMics[indSource, :], 
             waveNumber, 
             psfOutput)

    return psfOutput


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:]),
                 (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float32[:])],
                 '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _psf_Formulation1AkaClassic(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics]
        normalizeFactor = distSourcesToArrayCenter[cntSources] / nMics
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:]),
                 (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float32[:])],
                 '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _psf_Formulation2AkaInverse(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] * distGridToAllMics[cntMics]
        normalizeFactor = distSourcesToArrayCenter[cntSources] / distGridToArrayCenter[0] / nMics
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real  
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)  


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:]),
                 (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float32[:])],
                 '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _psf_Formulation3AkaTrueLevel(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        helpNormalizeGrid = 0.0
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] / distGridToAllMics[cntMics]
            helpNormalizeGrid += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])
        normalizeFactor = distSourcesToArrayCenter[cntSources] / distGridToArrayCenter[0] / helpNormalizeGrid
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor)


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:]),
                 (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float32[:])],
                 '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _psf_Formulation4AkaTrueLocation(distGridToArrayCenter, distGridToAllMics, distSourcesToArrayCenter, distSourcesToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntSources in range(len(distSourcesToArrayCenter)):
        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
        scalarProd = 0.0 + 0.0j
        helpNormalizeGrid = 0.0
        for cntMics in range(nMics):
            expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distSourcesToAllMics[cntSources, cntMics]))
            scalarProd += (np.cos(expArg) - 1j * np.sin(expArg)) / distSourcesToAllMics[cntSources, cntMics] / distGridToAllMics[cntMics]
            helpNormalizeGrid += 1.0 / (distGridToAllMics[cntMics] * distGridToAllMics[cntMics])
        normalizeFactor = distSourcesToArrayCenter[cntSources]
        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real
        result[cntSources] = scalarProdAbsSquared * (normalizeFactor * normalizeFactor) / nMics / helpNormalizeGrid

# CURRENTLY NOT NEEDED, AS CUSTOM PSF WILL BE CALCULATED IN fbeamform.SteeringVector WITH THE USE OF Trait transfer
#@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float64[:]),
#                 (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], nb.float32[:])],
#                 '(),(m),(s),(s,m),()->(s)', nopython=True, target=parallelOption, cache=cachedOption)
#def _psf_SpecificSteerVec(steerVec, steerVecSources, result):
#    nMics = len(steerVec)
#    for cntSources in range(steerVecSources.shape[0]):
#        # see bottom of information header of 'calcPointSpreadFunction' for infos on the PSF calculation and speed improvements.
#        scalarProd = 0.0 + 0.0j
#        for cntMics in range(nMics):
#            scalarProd += steerVec[cntMics].conjugate() * steerVecSources[cntSources, cntMics]
#        scalarProdAbsSquared = (scalarProd * scalarProd.conjugate()).real  
#        result[cntSources] = scalarProdAbsSquared


#%% Damas - Gauss Seidel
# Formerly known as 'gseidel'
@nb.guvectorize([#(nb.float32[:,:], nb.float32[:], nb.int64[:], nb.float64[:], nb.float32[:]), 
                 (nb.float64[:,:], nb.float64[:], nb.int64[:], nb.float64[:], nb.float64[:]),
                 #(nb.float32[:,:], nb.float64[:], nb.int64[:], nb.float64[:], nb.float64[:]),
                 #(nb.float64[:,:], nb.float32[:], nb.int64[:], nb.float64[:], nb.float32[:])
                 ], 
                 '(g,g),(g),(),()->(g)', nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def damasSolverGaussSeidel(A, dirtyMap, nIterations, relax, damasSolution):
    """ Solves the DAMAS inverse problem via modified gauss seidel.
    This is the original formulation from :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`.
    
    Parameters
    ----------
    A : float32/float64[nFreqs, nGridpoints, nGridpoints] (or float64[...])
        The PSF build matrix (see :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`)
    dirtyMap : float32/float64[nFreqs, nGridpoints] (or float64[...])
        The conventional beamformer map
    nIterations : int64[scalar] 
        number of Iterations the damas solver has to go through
    relax : int64[scalar] 
        relaxation parameter (=1.0 in :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`)
    damasSolution : float32/float64[nFreqs, nGridpoints] (or float64[...]) 
        starting solution
    
    Returns
    -------
    None : as damasSolution is overwritten with end result of the damas iterative solver.
    """
#    nGridPoints = len(dirtyMap)
#    for cntIter in range(nIterations[0]):
#        for cntGrid in range(nGridPoints):
#            solHelp = np.float32(0)
#            for cntGridHelp in range(cntGrid):  # lower sum
#                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
#            for cntGridHelp in range(cntGrid + 1, nGridPoints):  # upper sum
#                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
#            solHelp = (1 - relax[0]) * damasSolution[cntGrid] + relax[0] * (dirtyMap[cntGrid] - solHelp)
#            if solHelp > 0.0:
#                damasSolution[cntGrid] = solHelp
#            else:
#                damasSolution[cntGrid] = 0.0
    nGridPoints = len(dirtyMap)
    for cntIter in range(nIterations[0]):
        for cntGrid in range(nGridPoints):
            solHelp = 0.0
            for cntGridHelp in range(nGridPoints):  # full sum
                solHelp += A[cntGrid, cntGridHelp] * damasSolution[cntGridHelp]
            solHelp -= A[cntGrid, cntGrid] * damasSolution[cntGrid]
            solHelp = (1 - relax[0]) * damasSolution[cntGrid] + relax[0] * (dirtyMap[cntGrid] - solHelp)
            if solHelp > 0.0:
                damasSolution[cntGrid] = solHelp
            else:
                damasSolution[cntGrid] = 0.0


#%% Transfer - Function
def calcTransfer(distGridToArrayCenter, distGridToAllMics, waveNumber):
    """ Calculates the transfer functions between the various mics and gridpoints.
    
    Parameters
    ----------
    distGridToArrayCenter : float64[nGridpoints]
        Distance of all gridpoints to the center of sensor array
    distGridToAllMics : float64[nGridpoints, nMics]
        Distance of all gridpoints to all sensors of array
    waveNumber : complex128
        The wave number should be stored in the imag-part

    Returns
    -------
    The Transferfunctions in format complex128[nGridPoints, nMics].
    """
    nGridPoints, nMics = distGridToAllMics.shape[0], distGridToAllMics.shape[1]
    result = np.zeros((nGridPoints, nMics), np.complex128)
    # transfer routine: parallelized over Gridpoints
    _transferCoreFunc(distGridToArrayCenter, distGridToAllMics, np.array([waveNumber]), result)
    return result

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.complex128[:])], '(),(m),()->(m)', 
                nopython=True, target=parallelOption, cache=cachedOption, fastmath=fastOption)
def _transferCoreFunc(distGridToArrayCenter, distGridToAllMics, waveNumber, result):
    nMics = distGridToAllMics.shape[0]
    for cntMics in range(nMics):
        expArg = np.float32(waveNumber[0] * (distGridToAllMics[cntMics] - distGridToArrayCenter[0]))
        result[cntMics] = (np.cos(expArg) - 1j * np.sin(expArg)) * distGridToArrayCenter[0] / distGridToAllMics[cntMics]

