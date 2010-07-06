import beamfpy
print beamfpy.__file__

from os import path
import sys
from numpy import empty, clip, sqrt, arange, log10, sort, array, pi, zeros, \
hypot, cos, sin, linspace, hstack
from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, FiltOctave, BeamformerTime, TimePower, IntegratorSectorTime, \
PointSource, SineGenerator, Mixer


geo = '10.0'
freq = '150'

R = 2.5
Z = 4
#
loc0 = (-R,0.0,Z)
loc1 = (+R,0.0,Z)
loc2 = (0.0,+R,Z)
loc3 = (0.0,-R,Z)
loc4 = (R/sqrt(2),R/sqrt(2),Z)
loc5 = (R/sqrt(2),-R/sqrt(2),Z)
loc6 = (-R/sqrt(2),-R/sqrt(2),Z)
loc7 = (-R/sqrt(2),R/sqrt(2),Z)

m = MicGeom(from_file=path.join('beamfpy','xml','array92x.xml'))
r = 3
m.mpos_tot = array([(r*sin(2*pi*i+pi/4),r*cos(2*pi*i+pi/4),0) for i in linspace(0.0,1.0,27,False)]).T
xyz = m.mpos
c0 = 343.0
#===============================================================================
# inv_ch = []
# if geo == '64':
#     inv_ch = arange(64,92,1).tolist()
# elif geo == '28':
#     inv_ch = arange(0,64,1).tolist()
# elif geo == '29':
#     inv_ch = arange(0,64,1).tolist()
#     inv_ch.remove(8)
# elif geo == 'low':
#     inv_ch = [0,2,3,4,6,7,9,10,11,12,13,14,15,16,18,19,20,22,23,24,26,27,28,30,\
#     31,32,34,35,36,38,39,40,42,43,44,46,47,48,50,51,52,54,55,56,58,59,60,62,63]
# else:
#     rmax = float(geo)
#     dist = BeamformerTimeSq().env.r( 343.0, zeros((3,1)), m.mpos) 
#     inv_ch = arange(0,92,1).tolist()
#     for i in range(m.num_mics):
#         if dist[0,i]<rmax:
#             inv_ch.remove(i)
#     inv_ch = [0,2,4,6,7,9,10,11,12,13,14,15,16,18,20,22,23,24,26,28,30,\
#     31,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,60,62,63]\
#     + inv_ch
# m.invalid_channels = inv_ch
# 
#===============================================================================
s1 = SineGenerator(sample_freq=6144.0,numsamples=8192,freq=6144.0*3/128.0)
s2 = SineGenerator(sample_freq=6144.0,numsamples=8192,freq=6144.0*3/128.0,phase=pi)
p1 = PointSource(signal=s1, mpos=m,  loc=loc0)
p2 = PointSource(signal=s1, mpos=m,  loc=loc1)
p3 = PointSource(signal=s1, mpos=m,  loc=loc2)
p4 = PointSource(signal=s1, mpos=m,  loc=loc3)
p5 = PointSource(signal=s2, mpos=m,  loc=loc4)
p6 = PointSource(signal=s2, mpos=m,  loc=loc5)
p7 = PointSource(signal=s2, mpos=m,  loc=loc6)
p8 = PointSource(signal=s2, mpos=m,  loc=loc7)
t = Mixer(source = p1, sources = [p2, p3, p4, p5, p6, p7, p8 ])
print p1.sample_freq


f = PowerSpectra(window='Hanning',overlap='50%',block_size=128,ind_low=1,ind_high=30)
f.time_data = t
#g = RectGrid(x_min=-0.3,x_max=+0.3,y_min=-0.3,y_max=+0.3,z=loc[2],increment=0.05)
#g = RectGrid(x_min=-1.0,x_max=+1.0,y_min=-1.0,y_max=+1.0,z=loc[2],increment=0.0625)
#g = RectGrid(x_min=-1.0,x_max=+1.0,y_min=-1.0,y_max=+1.0,z=loc[2],increment=0.08)
#g = RectGrid(x_min=-2.0,x_max=+0.0,y_min=-1.0,y_max=+1.0,z=loc[2],increment=0.0625)
g = RectGrid(x_min=-3.0,x_max=+3.0,y_min=-3.0,y_max=+3.0,z=4,increment=0.15)
b = BeamformerBase(freq_data=f,grid=g,mpos=m,r_diag=False,c=343)

cfreq = float(freq)#1000

map = b.synthetic(cfreq,3)
#print map.shape
#print m.mpos[:,0]

ft = FiltFiltOctave(source = t)
ft.band = cfreq
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt,naverage=4096)
#for ft.band in (500,1000,2000,4000,8000):
#    for i in avgt.result(1):
#        print L_p(i[0,0])," dB"

bt = BeamformerTime(source = t,grid=g, mpos=m, c=343)
ft = FiltFiltOctave(source = bt)
ft.band = cfreq
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt)


fi = FiltFiltOctave(source = t)
fi.band = cfreq
fi.fraction = 'Third octave'
tr = Trajectory()
tr.points[0]=(1,0,0)
tr.points[4.0/30.0]=(-0.3,0,0)
#~ b = BeamformerTimeSqTraj(source = fi,grid=g, mpos=m, trajectory=tr)
bts = BeamformerTimeSq(source = fi,grid=g, mpos=m, r_diag=False,c=343)
bts.weights = 'none'
#bts.weights = 'constant power per unit radius'
#print bts.weights, bts.weights_

avg = TimeAverage(source = bts)
avg.naverage = 512
avgt.naverage = avg.naverage
cach = TimeCache( source = avg)
res = empty((t.numsamples/avg.naverage,g.size))
for fi.band in (cfreq,):#(500,1000,2000):#,4000,8000):
    j = 0
    for i in cach.result(4):
        s = i.shape[0]
        res[j:j+s] = i
        j += s
res = res[:j]
#print j
#g0 = RectGrid(x_min=-1.0,x_max=+1.0,y_min=-1.0,y_max=+1.0,z=loc[2],increment=0.0625)
#Inte = IntegratorSectorTime(source=cach, 
#                            sectors=[ (-0.5,-0.5, 0.5, 0.5),(-0.5,-0.5, 0.5, 0.5)], 
#                            grid=g0)
##                            grid=g)
#print L_p(array([i.copy() for i in Inte.result(1)]))#.reshape(-1,1).mean(0))
    


from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, psd
subplot(1,3,1)
f.time_data = t
map = b.synthetic(cfreq,1)
mx = L_p(map.max())
imshow(L_p(transpose(map)),vmax=mx, vmin=mx-10,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,2)
map1 = res.mean(axis=0).reshape(g.shape)
max1 = L_p(map1.max())
#print (L_p(map)+3>mx).nonzero()[0].shape,
#print (L_p(map1)+3>max1).nonzero()[0].shape
for x in (map,map1):
    b = (x[1:-1,1:-1]>x[0:-2,1:-1]) * (x[1:-1,1:-1]>x[2:,1:-1]) * (x[1:-1,1:-1]>x[1:-1,0:-2]) * (x[1:-1,1:-1]>x[1:-1,2:])
    x1 = x[1:-1,1:-1] * b / x.max()
    x3 = 10*log10(sort(x1.flat)[-6:])
    #print g.increment*sqrt(4/pi*(2*x>x.max()).nonzero()[0].shape[0]),x3

#imshow(L_p(transpose(map1[1:-1,1:-1] * b)),vmax=max1, vmin=mx-10,interpolation='nearest',extent=g.extend())
imshow(L_p(transpose(map1)),vmax=max1, vmin=max1-10,interpolation='nearest',extent=g.extend())
colorbar()
subplot(1,3,3)
map2 = clip(L_p(map)-L_p(map1),-20,20)
#print L_p(map.max())-L_p(map1.max())
imshow((transpose(map2)),vmax=10, vmin=-10,interpolation='nearest',extent=g.extend())
colorbar()
#print inv_ch
#figure(2)
#plot(m.mpos[0],m.mpos[1],'or')
#print m.mpos
show()


