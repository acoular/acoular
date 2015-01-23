# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------

"""
provisional speed test with open mp 

uses measured data in file 2008-05-16_11-36-00_468000.h5
calibration in file calib_06_05_2008.xml
microphone geometry in array_56.xml (part of acoular)

"""
from os import environ
environ['OMP_NUM_THREADS']='2'

# imports from acoular
import acoular
from acoular import td_dir, Calib, MicGeom, EigSpectra, \
RectGrid3D, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples

# other imports
from os import path

# files
datafile = path.join(td_dir,'2008-05-16_11-36-00_468000.h5')
calibfile = path.join(td_dir,'calib_06_05_2008.xml')
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')

#octave band of interest
cfreq = 4000

#===============================================================================
# first, we define the time samples using the MaskedTimeSamples class
# alternatively we could use the TimeSamples class that provides no masking
# of channels and samples
#===============================================================================
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
invalid = [1,7] # list of invalid channels (unwanted microphones etc.)
t1.invalid_channels = invalid 

#===============================================================================
# calibration is usually needed and can be set directly at the TimeSamples 
# object (preferred) or for frequency domain processing at the PowerSpectra 
# object (for backwards compatibility)
#===============================================================================
t1.calib = Calib(from_file=calibfile)

#===============================================================================
# the microphone geometry must have the same number of valid channels as the
# TimeSamples object has
#===============================================================================
m = MicGeom(from_file=micgeofile)
m.invalid_channels = invalid

#===============================================================================
# the grid for the beamforming map; the RectGrid3D class is used to produce a
# grid in a plane perpendicular to the array
# (the example grid is very coarse)
#===============================================================================
g = RectGrid3D(x_min=-0.6, x_max=+0.0, y_min=-0.4, y_max=0.4, \
    z_min=0.7, z_max=0.7, increment=0.005)

#===============================================================================
# for frequency domain methods, this provides the cross spectral matrix and its
# eigenvalues and eigenvectors, if only the matrix is needed then class 
# PowerSpectra can be used instead
#===============================================================================
f = EigSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               ind_low=4, ind_high=15) #to save computational effort, only
               # frequencies with index 1-30 are used


#===============================================================================
# different beamformers in frequency domain
#===============================================================================
# first, some simple delay and sum beamformers, 
# but with different steering vector types
#===============================================================================
bb0 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true level', cached = False) # this is the default
bs0 = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true level', cached=False, n=200)
be0 = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, n=-1, \
    steer='true level', cached=False) 
bo0 = BeamformerOrth(beamformer=be0, eva_list=range(38,54), cached=False)
    
from time import time
ti = time()
map = bs0.result[:]
print time()-ti, g.size
