# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2021, Acoular Development Team.
#------------------------------------------------------------------------------
"""
This file contains NUMBA accelerated functions for time-domain beamformers
"""
import numba as nb

cachedOption = True  # if True: saves the numba func as compiled func in sub directory
fastOption = True # fastmath options 


@nb.njit([(nb.float64[:,:], nb.int64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:])],
                cache=True, parallel=False, fastmath=True)
def _delayandsum(data, offsets, ifactor2, steeramp, out):
    """ Performs one time step of plain delay and sum
    
    **Note**: parallel could be set to true, but unless the number of gridpoints gets huge, it
    will be _slower_ in parallel mode
    
    Parameters
    ----------
    data : float64[nSamples, nMics] 
        The time history for all channels.
    offsets : int64[gridSize, nMics] 
        Indices for each grid point and each channel.
    ifactor2: float64[gridSize, nMics] 
        Second interpolation factor, the first one is computed internally.
    steeramp: float64[gridSize, nMics] 
        Amplitude factor from steering vector.
    
    Returns
    -------
    None : as the input out gets overwritten.
    """
    gridsize, numchannels = offsets.shape
    for gi in nb.prange(gridsize):
        out[gi] = 0
        for mi in range(numchannels):
            ind = offsets[gi,mi]
            out[gi] += (data[ind,mi] * (1-ifactor2[gi,mi]) \
                + data[ind+1,mi] * ifactor2[gi,mi]) * steeramp[gi,mi]  

@nb.njit([(nb.float64[:,:], nb.int64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64, nb.float64[:])],
                cache=True, parallel=False, fastmath=True)
def _delayandsum2(data, offsets, ifactor2, steeramp, dr, out):
    """ Performs one time step of delay and sum with output squared and optional autopower
    removal
    
    **Note**: parallel could be set to true, but unless the number of gridpoints gets huge, it
    will be _slower_ in parallel mode
    
    Parameters
    ----------
    data : float64[nSamples, nMics] 
        The time history for all channels.
    offsets : int64[gridSize, nMics] 
        Indices for each grid point and each channel.
    ifactor2: float64[gridSize, nMics] 
        Second interpolation factor, the first one is computed internally.
    steeramp: float64[gridSize, nMics] 
        Amplitude factor from steering vector.
    dr: float
        between 0 (no autopower removal) and 1.0 (full autopower removal)
        
    
    Returns
    -------
    None : as the input out gets overwritten.
    """
    gridsize, numchannels = offsets.shape
    for gi in nb.prange(gridsize):
        out[gi] = 0
        autopower = 0
        for mi in range(numchannels):
            ind = offsets[gi,mi]
            r = (data[ind,mi] * (1-ifactor2[gi,mi]) \
                + data[ind+1,mi] * ifactor2[gi,mi]) * steeramp[gi,mi]
            out[gi] += r
            autopower += r*r
        out[gi] = out[gi]*out[gi] - dr * autopower
        if out[gi]<1e-100:
            out[gi] = 1e-100