#! /usr/bin/env python
from acoular import *
from matplotlib import use
use('wx')
from pylab import *
from numpy import *
from os import path

ti = nidaq_import()
m = MicGeom(from_file=path.join( path.split(acoular.__file__)[0],'xml','array92x.xml'))
m.configure_traits()
ti.configure_traits()
ti.numsamples = int(ti.sample_freq/10.0)

nc = ti.numchannels
#~ nc = 64
if nc != m.num_mics:
    raise ValueError("Mikrofonanzahl passt nicht")
ti.get_single()

fig = figure()
#~ ax = fig.add_subplot(111)
#~ ax.set_title("Press 'a' to add stop")
# a single point
x = m.mpos[0]
y = m.mpos[1]
c = ones_like(x)*60.0
subplot(121)
sc = scatter(x,y,s=50,c=c, vmin=40, vmax=100)
axis('equal')
colorbar()
subplot(122)
pl0, = plot(c,ls='steps-mid',lw=2)
n = 10# zeitfenster fuer mittelung
mw = zeros((10,nc),'d')
cal = ones(nc,'d')*50
pl1, = plot(cal.copy(),ls='steps-mid',lw=2)
ylim(40,100)
last_calib = -1
            
def update(event):
    global last_calib
    d = ti.get_single()
    mw[1:]=mw[0:9]
    mw[0] = L_p(d.std(0))
    #~ mw[0] = random.rand(nc)*60+40
    data = mw.mean(0)
    st = mw.std(0)
    a = sc.get_array()
    sc.set_array(mw[0].copy())
    if st[data.argmax()]<1 and data.max()>90.0:
        dmax=data.argmax()
        if last_calib!=dmax:
            print dmax, data.max()
            cal[dmax]=data.max()
            pl1.set_ydata(cal.copy())
            last_calib=dmax
            savetxt('calib_helper.out',cal,'%f')
    else:
        last_calib=-1
        #~ pl1.draw()
    pl0.set_ydata(data)
    fig.canvas.draw()
    

import wx
id = wx.NewId()
actor = fig.canvas.manager.frame
timer = wx.Timer(actor, id=id)
timer.Start(200)
wx.EVT_TIMER(actor, id, update)

show()
print "ende"
