import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, RectGrid, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
TimeCache, FiltOctave, BeamformerTime, TimePower, BeamformerTimeSqTraj, \
WriteWAV, IntegratorSectorTime
from numpy import empty, loadtxt, arange, where, array
from os import path
import sys

print sys.argv

if sys.argv[1]=='1':
    h5 = "2009-10-10_12-51-48_906000.h5"
    richtung = "lr"
elif sys.argv[1]=='2':
    h5 = "2009-10-10_12-56-33_187000.h5"
    richtung = "lr"
elif sys.argv[1]=='3':
    h5 = "2009-09-30_13-02-29_890000.h5"
    richtung = "rl"
traj = loadtxt(path.join(td_dir,path.splitext(h5)[0]+'.traj'))

tr = Trajectory()

bildfreq = 30.0
bild = 0
for bildnr,x,y,z in traj:
#    print x
    if ((-0.5 < x < 1.0) and richtung == "rl") or\
            ((-1.0 < x < 0.5) and richtung == "lr"):
#    if (-1.0 < x < 1.0):
        if bild==0:
            bild=int(bildnr)
        print -x
        tr.points[(bildnr-bild)/bildfreq]=(-x,y,z-1.0)
        stopbild = int(bildnr)+1
        
print bild, stopbild, 2.0/((stopbild-bild)/bildfreq)

td=MaskedTimeSamples()
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


td.invalid_channels = inv_ch
m.invalid_channels = inv_ch
g = RectGrid(x_min=-1,x_max=1.0,y_min=-1,y_max=1,z=1.0,increment=0.0625)
#g = RectGrid(x_min=-1,x_max=1.0,y_min=-1,y_max=1,z=1.0,increment=0.125)
#g = RectGrid(x_min=-2,x_max=2.0,y_min=-2,y_max=2,z=1.0,increment=0.25)
td.name=path.join(td_dir,h5)
cal=Calib(from_file=path.join(td_dir,'calib92x20091012_64-6_28-3_inverted.xml'))
print cal.num_mics
td.calib = cal
td.start = (bild)*2048
td.stop= (stopbild)*2048
#ww = WriteWAV(source = td)
##ww.channels = [70,84]
#ww.channels = [0,]
#ww.save()
fi = FiltFiltOctave(source = td)#, fraction='Third octave')
fi.band = float(sys.argv[3])#4000
#b = BeamformerTimeSqTraj(source = fi,grid=g, mpos=m, trajectory=tr)
b = BeamformerTimeSq(source = fi,grid=g, mpos=m)
if sys.argv[4]=='w':
    b.weights = 'constant power per unit area'
    
avg = TimeAverage(source = b)
avg.naverage=512
cach = TimeCache( source = avg)
r = empty((1,td.numsamples/avg.naverage,g.size))
k = 0
#for fi.band in (2000,):
#for fi.band in (500,1000,2000,):
#~ for fi.band in (1000,2000,4000,):
#~ for fi.band in (2000,4000,8000,):
j = 0
for i in cach.result(4):
    s = i.shape[0]
#    print r[k,j:j+s].shape,i.shape
    r[k,j:j+s] = i
    j += s
    print j
k += 1
r=r[:,:j]
#~ r[:,0]  = r.mean(1)
sh = g.shape

Inte = IntegratorSectorTime(source=cach, 
                            sectors=[ array((-0.5,-0.5, 0.5, 0.5)),
                            (-0,-0.5, 0.5, 0.5),(-0.5,-0.5, 0.0, 0.5)], 
                            grid=g)
#for i in Inte.result(4):
#    print L_p(i)

from pylab import *
print cach.sample_freq
fignr = 1
for res in r:
    f = figure(fignr)
    gtr = tr.traj(1/cach.sample_freq)
    gin = Inte.result(1)
    a = f.add_subplot(6,6,36)
    map = L_p(res[:16].mean(axis=0))
    mx = map.max()
    mn = mx -10
    im = a.imshow(map.reshape(sh).T,vmin=mn,vmax=mx,extent=g.extend())
    colorbar(im,ax=a)
    plnr = 1
    for map in res:
        if plnr==6*6:
            break
        try:
            x,y,z = gtr.next()
        except:
            pass
        r2 = x*x + y*y + (z+1)*(z+1)
#        print x,y,z+1,sqrt(r2)
        a = f.add_subplot(6,6,plnr)
        mx1 = map.max()
#        map1 = where(map>mx1/10,map,0)
#        mx = 25
#        mn = mx-15
        mx1 = L_p(mx1)
        print x, y, sqrt(r2), L_p(gin.next())
        im = a.imshow(L_p(r2*map.reshape(sh).T),vmin=mn,vmax=mx,
        extent=g.extend(),interpolation='nearest')
        x1,y1,x2,y2 = Inte.sectors[0]
        a.plot((x1,x2,x2,x1,x1),(y1,y1,y2,y2,y1))
        a.plot((x,),(y,),'or')
        a.set_xticks([])
        a.set_yticks([])
        a.grid(b=True)
        plnr += 1
    fignr += 1
#figure(fignr)
#x = m.mpos[0]
#y = m.mpos[1]
#scatter(x,y)


#f=figure()
#a=f.add_subplot(211)
#mx = r[0].max(-1).max()
#mn = mx-30#r[0].min()
#a.plot(clip(r[0].max(-1),mn,mx))
#im={}
#for k in range(3):
#    a=f.add_subplot(2,3,4+k)
#    mx = r[k].max()
#    mn = mx-20#r[0].min()
#    im[k] = a.imshow(r[k,0].reshape(sh).T,vmin=mn,vmax=mx,extent=g.extend())
#    a.grid(b=True)
#    colorbar(im[k],ax=a)
#def update(y):
#    kc = y.GetKeyCode()
#    if kc == wx.WXK_RIGHT:
#        update.i += 1
#    if kc == wx.WXK_LEFT:
#        update.i -= 1              
#    if update.i==r.shape[1]:
#        update.i=0
#    print update.i
#    for k in range(3):
#        im[k].set_array(r[k,update.i].reshape(sh).T)
#    f.canvas.draw()
#update.i=0
#import wx
#wx.EVT_KEY_DOWN(wx.GetApp(), update)
name = '../'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'.png'
#savefig(name)
show()
