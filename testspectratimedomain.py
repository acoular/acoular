import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, synthetic, TimePower
from numpy import empty, array, random
from os import path
from enthought.traits.api import HasPrivateTraits, Float, Int

t1 = MaskedTimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
t1.stop = 16384
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

#t1 = TestSamples()
# Kalibrierdaten
cal = Calib(from_file=path.join(td_dir,'calib_06_05_2008.xml'))

# Mikrofongeometrie
m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))

f = PowerSpectra(window='Hanning',overlap='50%',block_size=2048)#,ind_low=1,ind_high=30)

f.time_data = t1


cfreq = 4000

fi = FiltFiltOctave(source = t1)
#fi.fraction = 'Third octave'
fi.band = cfreq
sq = TimePower(source = fi)
avg = TimeAverage(source = sq)
avg.naverage=1024
#cach = TimeCache( source = avg)
res = empty((t1.numsamples/avg.naverage,t1.numchannels))
#res = empty((t1.numsamples,t1.numchannels))
for fi.band in (cfreq,):#(500,1000,2000):#,4000,8000):
    j = 0
    for i in avg.result(4):
    #for i in t1.result(128):
        s = i.shape[0]
        res[j:j+s] = i
        j += s

ch = 20

levelfft = synthetic(f.csm[:,ch,ch],f.fftfreq(),array((cfreq,)),fi.fraction_)

#print res[:,0].std()**2, f.csm[1:,0,0].sum(), f.csm[0,0,0]
print res[:,ch].mean(), levelfft, f.csm[1:,ch,ch].sum(), f.csm[0,ch,ch]

th = empty((t1.numsamples,t1.numchannels))
j = 0
for i in t1.result(1024):
    s = i.shape[0]
    th[j:j+s] = i
    j += s
    
th1 = empty((t1.numsamples,t1.numchannels))
j = 0
for i in fi.result(1024):
    s = i.shape[0]
    th1[j:j+s] = i
    j += s    

print th[:,ch].std()**2, th[:,ch].mean()

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, psd
psd(th[:,ch],Fs=t1.sample_freq)
psd(th1[:,ch],Fs=t1.sample_freq)
show()


