import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, synthetic, TimePower, WriteWAV
from numpy import empty, array, random
from os import path
from enthought.traits.api import HasPrivateTraits, Float, Int

t1 = MaskedTimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
t1.stop = 5*51200
class TestSamples( TimeSamples ):
    
    sample_freq = Float(51200.0,
        desc="sampling frequency")
        
    numchannels = Int(10)
    
    numsamples = Int(16384)
        
    def result(self,N=128):
        i = 0
        random.seed(0)
        while i<self.numsamples:
            yield random.normal(size=(N,self.numchannels))
            i += N
        #while i<N*64:
         #   yield sin(2*pi*3000.0/61440.0*arange(i,i+N))[:,newaxis]*ones((N,10))
          #  i += N

ww = WriteWAV(source = t1)
ww.channels = [1,54]
ww.save()
