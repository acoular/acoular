#===============================================================================
# test example for timebeamformers with multiple sources derived from example2.py
#===============================================================================

from __future__ import print_function

import acoular
print(acoular.__file__)

from os import path
import sys
from numpy import empty, clip, sqrt, arange, log10, sort, array, pi, zeros, \
hypot, cos, sin, linspace, hstack, cross, dot, newaxis
from numpy.linalg import norm
from acoular import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
BeamformerTimeSqTraj, \
TimeCache, FiltOctave, BeamformerTime, TimePower, IntegratorSectorTime, \
PointSource, MovingPointSource, SineGenerator, WNoiseGenerator, Mixer, WriteWAV, \
PointSourceDipole, PNoiseGenerator,UncorrelatedNoiseSource, SourceMixer,\
WriteH5,BeamformerTimeTraj,UniformFlowEnvironment, FlowField,GeneralFlowEnvironment

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, \
psd, axis, xlim, ylim, title, suptitle

#===============================================================================
# some important definitions
#===============================================================================

freq = 6144.0*3/128.0 # frequency of interest (114 Hz)
sfreq = 6144.0/2 # sampling frequency (3072 Hz)
c0 = 343.0 # speed of sound
r = 3.0 # array radius
R = 2.5 # radius of source trajectory
Z = 4 # distance of source trajectory from 
#rps = 15.0/60. # revolutions per second
rps= 1
U = 1.0 # total number of revolutions

#===============================================================================
# construct the trajectory for the source
#===============================================================================

tr1 = Trajectory()
tmax = U/rps
delta_t = 1./rps/16.0 # 16 steps per revolution
for t in arange(0, tmax*1.001, delta_t):
    i = t* rps * 2 * pi #angle
    # define points for trajectory spline
    tr1.points[t] = (R*cos(i), R*sin(i), Z) # anti-clockwise rotation

#===============================================================================
# define circular microphone array and load other array geometries
#===============================================================================

m = MicGeom()
# set 28 microphone positions
m.mpos_tot = array([(r*sin(2*pi*i+pi/4), r*cos(2*pi*i+pi/4), 0) \
    for i in linspace(0.0, 1.0, 28, False)]).T

mg_file = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
mg = MicGeom(from_file=mg_file)
    
#===============================================================================
# define the different signals
#===============================================================================

if sys.version_info > (3,):
     long = int
nsamples = long(sfreq*tmax)

n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples)
s1 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq)
p1 = PNoiseGenerator(seed = 1, numsamples=nsamples)

#===============================================================================
# define environment and grid
#===============================================================================

ufe = UniformFlowEnvironment(ma=0.5,fdv=(0,1,0))
g = RectGrid(x_min=-3.0, x_max=+3.0, y_min=-3.0, y_max=+3.0, z=Z, increment=0.2)

#===============================================================================
# define some sources
#===============================================================================

mp = MovingPointSource(signal=s1, env=ufe, mpos=m, trajectory=tr1)
ps = PointSource(signal=s1, mpos=m,  loc=(0,R,Z))
pd = PointSourceDipole(signal=s1, mpos=m, direction=(0.5,0,0),  loc=(0,-2,Z))
un = UncorrelatedNoiseSource(signal=n1, mpos=mg)
mix = SourceMixer(sources=[ps,pd])
     
#===============================================================================
# 3rd Octave Filter 
#===============================================================================

fi = FiltFiltOctave(source=un, band=freq, fraction='Third octave')

#==============================================================================
#write to H5 File
#==============================================================================

# # write Signal to wav file
# ww = WriteWAV(source = p0)
# ww.channels = [0,27]
# ww.save()

# write H5 File
h5f1 = WriteH5(name='ps', source=ps).save()

#==============================================================================
#load H5 File
#==============================================================================

#load from h5 File
mt1 = MaskedTimeSamples(name='ps')

#==============================================================================
#fixed focus time domain beamforming point source
#==============================================================================

bt_ps = BeamformerTime(source=ps, grid=g, mpos=m, c=c0)
bt_ps_h5 = BeamformerTime(source=mt1, grid=g, mpos=m, c=c0)
btSq_ps = BeamformerTimeSq(source=ps, grid=g, mpos=m, r_diag=True, c=c0)
btSq_ps_h5 = BeamformerTimeSq(source=mt1, grid=g, mpos=m, r_diag=True, c=c0)

avgt_ps = TimeAverage(source=bt_ps, naverage=int(sfreq*tmax/2))
avgt_psSq = TimeAverage(source=btSq_ps, naverage=int(sfreq*tmax/2)) 
avgt_ps_h5 = TimeAverage(source=bt_ps_h5, naverage=int(sfreq*tmax/2))
avgt_psSq_h5 = TimeAverage(source=btSq_ps_h5, naverage=int(sfreq*tmax/2))

#==============================================================================
#plot point source
#==============================================================================

figure(1)
for res in avgt_ps.result(1):
     res_ps = res[0].reshape(g.shape)
     subplot(2,2,1)
     mx = L_p(res_ps.max())
     imshow(L_p(transpose(res_ps)), vmax=mx, vmin=mx-10, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps bft avgt")
     colorbar()
for res in avgt_ps_h5.result(1):
     res_ps = res[0].reshape(g.shape)
     subplot(2,2,2)
     mx = L_p(res_ps.max())
     imshow(L_p(transpose(res_ps)), vmax=mx, vmin=mx-10, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps bft avgt h5")
     colorbar()
for res in avgt_psSq.result(1):
     res_ps = res[0].reshape(g.shape)
     subplot(2,2,3)
     mx = L_p(res_ps.max())
     imshow(L_p(transpose(res_ps)), vmax=mx, vmin=mx-10, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps bftSq avgt")
     colorbar()
for res in avgt_psSq_h5.result(1):
     res_ps = res[0].reshape(g.shape)
     subplot(2,2,4)
     mx = L_p(res_ps.max())
     imshow(L_p(transpose(res_ps)), vmax=mx, vmin=mx-10, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps bftSq avgt h5")
     colorbar()

#==============================================================================
#fixed focus time domain beamforming mixed point source and dipol source
#==============================================================================

bt_mix = BeamformerTime(source=mix, grid=g, mpos=m, c=c0)
btSq_mix = BeamformerTimeSq(source=mix, grid=g, mpos=m, r_diag=True, c=c0)

tp_bt_mix = acoular.TimePower(source=bt_mix)
tp_btSq_mix = acoular.TimePower(source=btSq_mix)

avgt_mix = TimeAverage(source=tp_bt_mix, naverage=int(sfreq*tmax/2))
avgt_mixSq = TimeAverage(source=tp_btSq_mix, naverage=int(sfreq*tmax/2)) 

#==============================================================================
#plot mix source
#==============================================================================

figure(2)
for res in avgt_mix.result(1):
     res_mix = res[0].reshape(g.shape)
     subplot(1,2,1)
     mx = L_p(res_mix.max())
     imshow(L_p(transpose(res_mix)), vmax=mx, vmin=mx-20, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps and dp")
     colorbar()
for res in avgt_mixSq.result(1):
     res_mix = res[0].reshape(g.shape)
     subplot(1,2,2)
     mx = L_p(res_mix.max())
     imshow(L_p(transpose(res_mix)), vmax=mx, vmin=mx-20, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("ps and dp bftSq")
     colorbar()


#==============================================================================
#fixed/moving focus time domain beamforming moving point source with and 
#without flow
#==============================================================================

bt_mp = BeamformerTime(source=mp, grid=g, mpos=m, c=c0)
btTr_mp = BeamformerTimeTraj(source=mp, grid=g, mpos=m,trajectory=tr1,rvec = array((0,0,1.0)))
btSqTr_mp = BeamformerTimeSqTraj(source=mp, grid=g, mpos=m,trajectory=tr1,rvec = array((0,0,1.0)))

tp_bt_mp = acoular.TimePower(source=bt_mp)
tp_btTr_mp = acoular.TimePower(source=btTr_mp)
tp_btSqTr_mp = acoular.TimePower(source=btSqTr_mp)

avgt_bt_mp = TimeAverage(source=tp_bt_mp, naverage=int(sfreq*tmax/2))
avgt_btTr_mp = TimeAverage(source=tp_btTr_mp, naverage=int(sfreq*tmax/2))
avgt_btSqTr_mp = TimeAverage(source=tp_btSqTr_mp, naverage=int(sfreq*tmax/2))

#==============================================================================
#plot moving point source with flow
#==============================================================================

figure(3)
for res in avgt_bt_mp.result(1):
     res_mp = res[0].reshape(g.shape)
     subplot(1,3,1)
     mx = L_p(res_mp.max())
     imshow(L_p(transpose(res_mp)), vmax=mx, vmin=mx-20, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("mp bt fixed")
     colorbar()
for res in avgt_btTr_mp.result(1):
     res_mp = res[0].reshape(g.shape)
     subplot(1,3,2)
     mx = L_p(res_mp.max())
     imshow(L_p(transpose(res_mp)), vmax=mx, vmin=mx-20, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("mp bt moving")
     colorbar()
for res in avgt_btSqTr_mp.result(1):
     res_mp = res[0].reshape(g.shape)
     subplot(1,3,3)
     mx = L_p(res_mp.max())
     imshow(L_p(transpose(res_mp)), vmax=mx, vmin=mx-20, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("mp btSq moving")
     colorbar()
     
#==============================================================================
#time domain beamforming noise
#==============================================================================

bt_un = BeamformerTime(source=fi, grid=g, mpos=m, c=c0)
avgt_un = TimeAverage(source=bt_un, naverage=int(sfreq*tmax/2))

cacht = acoular.TimeCache( source = avgt_un) # cache to prevent recalculation


#==============================================================================
#plot moving point source with flow
#==============================================================================

figure(4)
for res in cacht.result(1):
     res_un = res[0].reshape(g.shape)
     subplot(1,1,1)
     mx = L_p(res_un.max())
     imshow(L_p(transpose(res_un)), vmax=mx, vmin=mx-10, interpolation='nearest',\
            extent=g.extend(), origin='lower')
     title("bt uncorr noise")
     colorbar()
#==============================================================================
