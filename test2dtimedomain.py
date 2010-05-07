import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, FiltOctave, BeamformerTime, TimePower
from numpy import empty, clip
from os import path

t = TimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
t1 = MaskedTimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))
t1.stop = 18000
# Kalibrierdaten
cal = Calib(from_file=path.join(td_dir,'calib_06_05_2008.xml'))

# Mikrofongeometrie
m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))
m1 = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56_minus_3_7_9.xml'))

f = PowerSpectra(window='Hanning',overlap='50%',block_size=128,ind_low=1,ind_high=30)
f.time_data = t1
g = RectGrid(x_min=-0.6,x_max=-0.0,y_min=-0.3,y_max=0.3,z=0.68,increment=0.05)

b = BeamformerBase(freq_data=f,grid=g,mpos=m,r_diag=False,c=346.04)

cfreq = 4000

map = b.synthetic(cfreq,1)
print map.shape
print map[0,0]

bt = BeamformerTime(source = t1,grid=g, mpos=m, c=346.04)
ft = FiltFiltOctave(source = bt)
ft.band = cfreq
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt)


fi = FiltFiltOctave(source = t1)
fi.band = cfreq
tr = Trajectory()
tr.points[0]=(1,0,0)
tr.points[4.0/30.0]=(-0.3,0,0)
#~ b = BeamformerTimeSqTraj(source = fi,grid=g, mpos=m, trajectory=tr)
bts = BeamformerTimeSq(source = fi,grid=g, mpos=m, r_diag=False,c=346.04)
avg = TimeAverage(source = bts)
avg.naverage = 1024
avgt.naverage = avg.naverage
cach = TimeCache( source = avg)
res = empty((t1.numsamples/avg.naverage,g.size))
for fi.band in (cfreq,):#(500,1000,2000):#,4000,8000):
    j = 0
    for i in cach.result(4):
        s = i.shape[0]
        res[j:j+s] = i
        j += s

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, psd
subplot(1,3,1)
f.time_data = t1
map = b.synthetic(cfreq,1)
mx = L_p(map.max())
imshow(L_p(transpose(map)),vmax=mx, vmin=mx-15,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,2)
map1 = res.mean(axis=0).reshape(g.shape)
imshow(L_p(transpose(map1)),vmax=mx, vmin=mx-15,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,3)
map2 = clip(L_p(map)-L_p(map1),-20,20)
imshow((transpose(map2)),vmax=3, vmin=-3,interpolation='nearest',extent=g.extend())
colorbar()
show()


