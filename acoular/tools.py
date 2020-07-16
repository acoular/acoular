# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2021, Acoular Development Team.
#------------------------------------------------------------------------------
"""
Implements tools for Acoular.

.. autosummary::
    :toctree: generated/

    barspectrum
    bardata
"""

from traits.api import HasStrictTraits
from numpy import array, concatenate, newaxis, where
from numpy.ma import masked_where
from .spectra import synthetic


def barspectrum(data, fftfreqs, num = 3, bar = True, xoffset = 0.0):
    """
    Returns synthesized frequency band values of spectral data to be plotted
    as bar graph with the matlpotlib plot command.
    
   
    Parameters
    ----------
    data : array of floats
        The spectral data (sound pressures in Pa) in an array with one value 
        per frequency line.
    fftfreqs : array of floats
        Discrete frequencies from FFT.
    num : integer
        Controls the width of the frequency bands considered; defaults to
        3 (third-octave band).
    bar : bool
        If True, returns bar-like curve. If False, normal plot (direct 
        line between data points) is returned.
    xoffset : float
        If bar is True, offset of the perpendicular line (helpful if 
        plotting several curves above each other). 
        
        ===  =====================
        num  frequency band width
        ===  =====================
        1    octave band
        3    third-octave band
        ===  =====================

    Returns
    -------
    (flulist, plist, fc)
    flulist : array of floats
        Lower/upper band frequencies in plottable format.
    plist : array of floats
        Corresponding synthesized frequency band values in plottable format.
    fc : array of floats
        Evaluated band center frequencies.
    """
    
    if num not in [1,3]:
        print('Only octave and third-octave bands supported at this moment.')
        return (0,0,0)
        

    # preferred center freqs after din en iso 266 for third-octave bands
    fcbase = array([31.5,40,50,63,80,100,125,160,200,250])
    # DIN band center frequencies from 31.5 Hz to 25 kHz
    fc = concatenate((fcbase, fcbase*10., fcbase[:]*100.))[::(3//num)]
    
    
    # exponent for band width calculation
    ep = 1. / (2.*num)
    
    # lowest and highest possible center frequencies 
    # for chosen band and sampling frequency
    f_low = fftfreqs[1]*2**ep
    f_high = fftfreqs[-1]*2**-ep
    # get possible index range
    if fc[0] >= f_low:
        i_low = 0
    else:
        i_low = where(fc < f_low)[0][-1]
    
    if fc[-1] <= f_high:
        i_high = fc.shape[0]
    else:
        i_high = where(fc > f_high)[0][0]
        
    # synthesize sound pressure values
    p = array([ synthetic(data, fftfreqs, list(fc[i_low:i_high]), num) ])
 
    if bar:
        # upper and lower band borders
        flu = concatenate(( fc[i_low:i_low+1]*2**-ep,
                            ( fc[i_low:i_high-1]*2**ep + fc[i_low+1:i_high]*2**-ep ) / 2.,
                            fc[i_high-1:i_high]*2**ep ))
        # band borders as coordinates for bar plotting
        flulist = 2**(2*xoffset*ep) * (array([1,1])[:,newaxis]*flu[newaxis,:]).T.reshape(-1)[1:-1]   
        # sound pressures as list for bar plotting
        plist = (array([1,1])[:,newaxis]*p[newaxis,:]).T.reshape(-1)
    else:
        flulist = fc[i_low:i_high]
        plist = p[0,:]
    #print(flulist.shape, plist.shape)
    return (flulist, plist, fc[i_low:i_high])
    


def bardata(data, fc, num=3, bar = True, xoffset = 0.0, masked = -360):
    """
    Returns data to be plotted
    as bar graph with the matlpotlib plot command.
    
   
    Parameters
    ----------
    data : array of floats
        The spectral data 
    fc : array of floats
        Band center frequencies
    bar : bool
        If True, returns bar-like curve. If False, normal plot (direct 
        line between data points) is returned.
    xoffset : float
        If bar is True, offset of the perpendicular line (helpful if 
        plotting several curves above each other). 
        
        ===  =====================
        num  frequency band width
        ===  =====================
        1    octave band
        3    third-octave band
        ===  =====================

    Returns
    -------
    (flulist, plist)
    flulist : array of floats
        Lower/upper band frequencies in plottable format.
    plist : array of floats
        Corresponding values in plottable format.
    """
    ep = 1. / (2.*num)
    
    if bar:
        # upper and lower band borders
        flu = concatenate(( fc[:1]*2**-ep,
                            ( fc[:-1]*2**ep + fc[1:]*2**-ep ) / 2.,
                            fc[-1:]*2**ep ))
        # band borders as coordinates for bar plotting
        flulist = 2**(xoffset*1./num) * (array([1,1])[:,newaxis]*flu[newaxis,:]).T.reshape(-1)[1:-1]   
        # sound pressures as list for bar plotting
        plist = (array([1,1])[:,newaxis] * data[newaxis,:]).T.reshape(-1)
    else:
        flulist = fc
        plist = data
    #print(flulist.shape, plist.shape)
    if masked > -360:
        plist = masked_where(plist <= masked, plist)
    return (flulist, plist)

