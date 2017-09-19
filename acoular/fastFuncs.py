#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains all the functionalities which are very expansive, regarding
computational costs. All functionalities are optimized via NUMBA.
"""
import numpy as np
import numba as nb

cachedOption = True  # if True: saves the numba func as compiled func in sub directory
parallelOption = 'parallel'  # if numba.guvectorize is used: 'CPU' for single threading; 'parallel' for multithreading; 'cuda' for calculating on GPU


# Formerly known as 'faverage'
@nb.njit(nb.complex128[:,:,:](nb.complex128[:,:,:], nb.complex128[:,:]), cache=cachedOption)
def calcCSM(csm, SpecAllMics):
    """ Adds a given spectrum to the Cross-Spectral-Matrix (CSM).
    Here only the upper triangular matrix of the CSM is calculated. After
    averaging over the various ensembles, the whole CSM is created via complex 
    conjugation transposing. This happens outside (in acoular.spectra). This method
    was called 'faverage' in earlier versions of acoular.
    
    Input
    -----
        ``csm`` ... complex128[nFreqs, nMics, nMics] --> the current CSM.
        
        ``SpecAllMics`` ...complex128[nFreqs, nMics] --> spectrum of the added ensemble at all Mics.
    
    Returns
    -------
        ``None`` ... as the input ``csm`` gets overwritten.
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

