from beamfpy import *
from pylab import *
from numpy import *
from os import path

td = TimeSamples()
td.name = 'calib'
ti = nidaq_import()
m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_64.xml'))
m.configure_traits()
ti.configure_traits()
ti.numsamples = int(ti.sample_freq/10.0)

nc = ti.numchannels
nc = 64
#~ if nc != m.num_mics:
	#~ raise ValueError("Mikrofonanzahl passt nicht")

fig = figure()
#~ ax = fig.add_subplot(111)
#~ ax.set_title("Press 'a' to add stop")
# a single point
x = m.mpos[0]
y = m.mpos[1]
c = ones_like(x)*60.0
subplot(121)
sc = scatter(x,y,s=100,c=c, vmin=40, vmax=100)
axis('equal')
colorbar()
subplot(122)
pl0, = plot(c,ls='steps-mid',lw=5)
n = 10# zeitfenster fuer mittelung
mw = zeros((10,nc),'d')
cal = ones(nc,'d')*50
pl1, = plot(cal.copy(),ls='steps-mid',lw=5)
ylim(40,100)
			
def update(event):
	#~ ti.get_data(td)
	#~ d = td.data[:]
	mw[1:]=mw[0:9]
	#~ mw[0] = L_p(d.std(0))
	mw[0] = random.rand(nc)*60+40
	data = mw.mean(0)
	st = mw.std(0)
	a = sc.get_array()
	sc.set_array(data)
	if st[data.argmax()]<10 and data.max()>80.0:
		print data.argmax(), data.max()
		cal[data.argmax()]=data.max()
		pl1.set_ydata(cal.copy())
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
