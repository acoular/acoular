import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra,\
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc
from beamfpy.timedomain import MaskedTimeSamples
from os import path



t = TimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
t1 = MaskedTimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
# Kalibrierdaten
cal = Calib(from_file=path.join(td_dir,'calib_06_05_2008.xml'))

# Mikrofongeometrie
m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))
m1 = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56_minus_3_7_9.xml'))

f = PowerSpectra(window='Hanning',overlap='50%',block_size=128,ind_low=15,ind_high=30)
f.time_data = t
print f.fftfreq()[5]
g = RectGrid(x_min=-0.6,x_max=-0.0,y_min=-0.3,y_max=0.3,z=0.68,increment=0.005)

b = BeamformerBase(freq_data=f,grid=g,mpos=m,r_diag=True,c=346.04)
#bc = BeamformerCapon(freq_data=f,grid=g,mpos=m,c=346)
#be = BeamformerEig(freq_data=f,grid=g,mpos=m,r_diag=True,c=346,n=50)
#bm = BeamformerMusic(freq_data=f,grid=g,mpos=m,r_diag=True,c=346)
#bd = BeamformerDamas(beamformer=b,n_iter=100)
#bo = BeamformerOrth(beamformer=be, eva_list=range(40,56))
#bs = BeamformerCleansc(freq_data=f,grid=g,mpos=m,r_diag=True,c=346)
map = b.synthetic(8000,1)
print map.shape
print map[0,0]

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure
subplot(1,3,1)
imshow(L_p(transpose(map)),vmin=30,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,2)
f.time_data = t1
map = b.synthetic(8000,1)
imshow(L_p(transpose(map)),vmin=30,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,3)
t1.stop = 512000
t1.invalid_channels = [3,7,9]
b.mpos = m1
map = b.synthetic(8000,1)
imshow(L_p(transpose(map)),vmin=30,interpolation='nearest',extent=g.extend())
colorbar()
#for i in range(2,10):
#    subplot(3,3,i)
#    bm.n=i-1+8
#    map = bm.synthetic(2000,0)
#    print i
#    imshow(L_p(transpose(map)),vmax=0,vmin=-20,interpolation='nearest',extent=g.extend())
#    colorbar()
#imshow(transpose(map))
#figure(2)
#plot(L_p(b.integrate((-0.3,-0.1,-0.1,0.1)))[f.ind_low:f.ind_high])
#plot(L_p(b.integrate((-0.24,0.2,-0.22,0.3)))[f.ind_low:f.ind_high])
show()


