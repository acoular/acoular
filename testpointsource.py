import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, FiltOctave, BeamformerTime, TimePower
from numpy import empty, clip, sqrt, arange, log10, sort
from os import path
import sys

from beamfpy.sources import PointSource

loc = (0.0,0.0,1.0)

m = MicGeom(from_file=path.join(td_dir,'array_92x_x3_9_y-1_7_mod.xml'))
c0 = 343.0
inv_ch=[]
if sys.argv[2]=='64':
    inv_ch = arange(64,92,1).tolist()
elif sys.argv[2]=='28':
    inv_ch = arange(0,64,1).tolist()
elif sys.argv[2]=='29':
    inv_ch = arange(0,64,1).tolist()
    inv_ch.remove(8)
elif sys.argv[2]=='low':
    inv_ch = [0,2,3,4,6,7,9,10,11,12,13,14,15,16,18,19,20,22,23,24,26,27,28,30,\
    31,32,34,35,36,38,39,40,42,43,44,46,47,48,50,51,52,54,55,56,58,59,60,62,63]
elif sys.argv[2]=='center':
    inv_ch = arange(0,92,1).tolist()
    inv_ch.remove(8)
m.invalid_channels = inv_ch

#m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))
t = PointSource(sample_freq=32000.0,numsamples=4096,mpos=m, loc=loc)
f = PowerSpectra(window='Hanning',overlap='50%',block_size=128,ind_low=1,ind_high=30)
f.time_data = t
g = RectGrid(x_min=-0.3,x_max=+0.3,y_min=-0.3,y_max=+0.3,z=loc[2],increment=0.05)
g = RectGrid(x_min=-1.0,x_max=+1.0,y_min=-1.0,y_max=+1.0,z=loc[2],increment=0.0625)
b = BeamformerBase(freq_data=f,grid=g,mpos=m,r_diag=False,c=343)

cfreq = 500

map = b.synthetic(cfreq,1)
print map.shape
print map[0,0]

ft = FiltFiltOctave(source = t)
ft.band = cfreq
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt,naverage=4096)
for i in avgt.result(1):
    print L_p(i)


bt = BeamformerTime(source = t,grid=g, mpos=m, c=343)
ft = FiltFiltOctave(source = bt)
ft.band = cfreq
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt)


fi = FiltFiltOctave(source = t)
fi.band = cfreq
tr = Trajectory()
tr.points[0]=(1,0,0)
tr.points[4.0/30.0]=(-0.3,0,0)
#~ b = BeamformerTimeSqTraj(source = fi,grid=g, mpos=m, trajectory=tr)
bts = BeamformerTimeSq(source = fi,grid=g, mpos=m, r_diag=False,c=346.04)
avg = TimeAverage(source = bts)
avg.naverage = 64
avgt.naverage = avg.naverage
cach = TimeCache( source = avgt)
res = empty((t.numsamples/avg.naverage,g.size))
for fi.band in (cfreq,):#(500,1000,2000):#,4000,8000):
    j = 0
    for i in cach.result(4):
        s = i.shape[0]
        res[j:j+s] = i
        j += s
res = res[:j]

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, psd
subplot(1,3,1)
f.time_data = t
map = b.synthetic(cfreq,1)
mx = L_p(map.max())
imshow(L_p(transpose(map)),vmax=mx, vmin=mx-15,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,2)
map1 = res.mean(axis=0).reshape(g.shape)
max1 = L_p(map1.max())
print (L_p(map)+3>mx).nonzero()[0].shape,
print (L_p(map1)+3>max1).nonzero()[0].shape
for x in (map,map1):
    b = (x[1:-1,1:-1]>x[0:-2,1:-1]) * (x[1:-1,1:-1]>x[2:,1:-1]) * (x[1:-1,1:-1]>x[1:-1,0:-2]) * (x[1:-1,1:-1]>x[1:-1,2:])
    x1 = x[1:-1,1:-1] * b / x.max()
    x3 = 10*log10(sort(x1.flat)[-2])
    print (2*x>x.max()).nonzero()[0].shape[0],x3

imshow(L_p(transpose(map1)),vmax=mx, vmin=mx-15,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,3)
map2 = clip(L_p(map)-L_p(map1),-20,20)
print L_p(map.max())-L_p(map1.max())
imshow((transpose(map2)),vmax=10, vmin=-10,interpolation='nearest',extent=g.extend())
colorbar()
show()


