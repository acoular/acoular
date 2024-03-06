# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""
Implements tools for Acoular.

.. autosummary::
    :toctree: generated/
    
    MetricEvaluator
    return_result
    spherical_hn1
    get_radiation_angles
    get_modes
    barspectrum
    bardata
"""

import acoular as ac
from numpy import array, concatenate, newaxis, where,arctan2,sqrt,pi,mod,zeros,\
    complex128, ones, inf, minimum, empty
from numpy.linalg import norm
from numpy.ma import masked_where
from .spectra import synthetic
from traits.api import Bool, CArray, HasPrivateTraits, Instance, Property, Any

from scipy.special import spherical_yn, spherical_jn, sph_harm
from scipy.spatial.distance import cdist
from copy import copy

class MetricEvaluator(HasPrivateTraits):
    """Evaluate the reconstruction performance of source mapping methods.

    This class can be used to calculate the following performance metrics
    according to Herold and Sarradj (2017):
    * Specific level error
    * Overall level error
    * Inverse level error
    """

    #: an array of shape=(nf,ng) containing the squared sound pressure data of the   
    #: source mapping. (nf: number of frequencies, ng: number of grid points)
    data = CArray(shape=(None,None),
        desc="Contains the calculated squared sound pressure values in Pa**2.")

    #: an array of shape=(nf,ns) containing the squared sound pressure data of the
    #: ground-truth sources. (nf: number of frequencies, ns: number of sources)
    target_data = CArray(shape=(None,None),
        desc="Contains the ground-truth squared sound pressure values in Pa**2.")

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations
    #: for the calculated source mapping data.
    grid = Instance(ac.Grid,
        desc="Grid instance that belongs to the calculated data")

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations
    #: for the ground-truth data.
    target_grid = Instance(ac.Grid,
        desc="Grid instance that belongs to the ground-truth data")

    #: sector type. Currently only circular sectors are supported.
    sector = Instance(ac.CircSector, default=ac.CircSector(),)

    #: if set True: use shrink integration area if two sources are closer
    #: than 2*r. The radius of the integration area is then set to half the
    #: distance between the two sources.
    adaptive_sector_size = Bool(True,
        desc="adaptive integration area")

    #: if set `True`, the same amplitude can be assigned to multiple targets if
    #: the integration area overlaps. If set `False`, the amplitude is assigned
    #: to the first target and the other targets are ignored.
    multi_assignment = Bool(True,
        desc="if set True, the same amplitude can be assigned to multiple targets if the integration area overlaps")

    #: returns the determined sector sizes for each ground-truth source position
    sectors = Property()

    def _validate_shapes(self):
        if self.data.shape[0] != self.target_data.shape[0]:
            raise ValueError("data and target_data must have the same number of frequencies!")
        if self.data.shape[1] != self.grid.size:
            raise ValueError("data and grid must have the same number of grid points!")
        if self.target_data.shape[1] != self.target_grid.size:
            raise ValueError("target_data and target_grid must have the same number of grid points!")

    def _get_sector_radii(self):
        ns = self.target_data.shape[1]
        radii = ones(ns)*self.sector.r
        if self.adaptive_sector_size:
            locs = self.target_grid.gpos.T
            intersrcdist = cdist(locs, locs)
            intersrcdist[intersrcdist == 0] = inf
            intersrcdist = intersrcdist.min(0)/2
            radii = minimum(radii,intersrcdist)
        return radii

    def _get_sectors(self):
        """Returns a list of CircSector objects for each target location."""
        r = self._get_sector_radii()
        ns = self.target_data.shape[1]
        sectors = []
        for i in range(ns):
            loc = self.target_grid.gpos[:,i]
            sector = copy(self.sector)
            sector.r = r[i]
            sector.x = loc[0]
            sector.y = loc[1]
            sectors.append(sector)
        return sectors

    def _integrate_sectors(self):
        """Integrates over target sectors.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated Pa**2 values for each sector
        """
        sectors = self.sectors
        results = empty(shape=self.target_data.shape)
        for f in range(self.target_data.shape[0]):
            data = self.data[f]
            for i in range(self.target_data.shape[1]):
                sector = sectors[i]
                results[f,i] = ac.integrate(data,self.grid,sector)
                if not self.multi_assignment:
                    indices = self.grid.subdomain(sector)
                    data[indices] = 0 # set values to zero (can not be assigned again)
        return results

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,)
        """
        self._validate_shapes()
        return ac.L_p(self.data.sum(axis=1)) - ac.L_p(self.target_data.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources
        """
        self._validate_shapes()
        sector_result = self._integrate_sectors()
        return ac.L_p(sector_result) - ac.L_p(self.target_data)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)
        """
        self._validate_shapes()
        sector_result = self._integrate_sectors()
        return ac.L_p(sector_result.sum(axis=1)) - ac.L_p(self.data.sum(axis=1))



def return_result(source, nmax=-1, num=128):
    """
    Collects the output from a 
    :meth:`SamplesGenerator.result()<acoular.tprocess.SamplesGenerator.result>`
    generator and returns an assembled array with all the data.
   
    Parameters
    ----------
    source: SamplesGenerator or derived object.
        This is the  :class:`SamplesGenerator<acoular.tprocess.SamplesGenerator>` data source.
    nmax: integer
        With this parameter, a maximum number of output samples can be set 
        (first dimension of array). If set to -1 (default), samples are 
        collected as long as the generator yields them.
    num : integer
        This parameter defines the size of the blocks that are fetched.
        Defaults to 128.
          
    Returns
    -------
    array of floats (number of samples, source.numchannels)
        Array that holds all the data.
    """
    resulter = (_.copy() for _ in source.result(num))
    
    if nmax > 0: 
        nblocks = (nmax-1) // num + 1
        return concatenate( 
                      list( res for _, res in 
                            zip(range(nblocks), 
                                resulter) ) )[:nmax]
    else:
        return concatenate(list(resulter))


def spherical_hn1(n,z,derivativearccos=False):
   """ Spherical Hankel Function of the First Kind 
   
   """
   return spherical_jn(n,z,derivative=False)+1j*spherical_yn(n,z,derivative=False) 

def get_radiation_angles(direction,mpos, sourceposition):
    """
    Returns azimuthal and elevation angles between the mics and the source 
    
    Parameters
    ----------
    direction : array of floats
        Spherical Harmonic orientation
    mpos : array of floats
        x, y, z position of microphones
    sourceposition : array of floats
        position of the source        
        
    Returns
    -------
    azi, ele : array of floats
        the angle between the mics and the source 
    """
    #direction of the Spherical Harmonics
    direc = array(direction, dtype = float)
    direc = direc/norm(direc)
    # distances
    source_to_mic_vecs = mpos-array(
        sourceposition).reshape((3, 1))
    source_to_mic_vecs[2] *= -1 # invert z-axis (acoular)    #-1
    # z-axis (acoular) -> y-axis (spherical)
    # y-axis (acoular) -> z-axis (spherical)
    #theta
    ele = arctan2(sqrt(source_to_mic_vecs[0]**2 + source_to_mic_vecs[2]**2),source_to_mic_vecs[1])
    ele +=arctan2(sqrt(direc[0]**2 + direc[2]**2), direc[1])  
    ele += pi*.5 # convert from [-pi/2, pi/2] to [0,pi] range
    #phi
    azi = arctan2(source_to_mic_vecs[2],source_to_mic_vecs[0]) 
    azi += arctan2(direc[2],direc[0]) 
    azi = mod(azi,2*pi)
    return azi, ele

def get_modes(lOrder, direction, mpos , sourceposition = array([0,0,0])):
    """
    Returns Spherical Harmonic Radiation Pattern at the Microphones
      
    Parameters
    ----------
    lOrder : int
        Maximal order of spherical harmonic
    direction : array of floats
        Spherical Harmonic orientation
    mpos : array of floats
        x, y, z position of microphones
    sourceposition : array of floats
        position of the source

    Returns
    -------
    modes : array of floats
        the radiation values at each microphone for each mode
    """
    azi, ele = get_radiation_angles(direction,mpos,sourceposition) # angles between source and mics 
    modes = zeros((azi.shape[0], (lOrder+1)**2), dtype=complex128)
    i = 0
    for l in range(lOrder+1):
        for m in range(-l, l+1):
            modes[:, i] = sph_harm(m, l, azi, ele)
            if m<0:
                modes[:, i]=modes[:, i].conj()*1j             
            i += 1
    return modes

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

