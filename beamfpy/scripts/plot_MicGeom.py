from beamfpy import *
from pylab import *
 
m=MicGeom()
m.configure_traits()
plot(m.mpos[0],m.mpos[1],'or')
figure(1,(4,4))
map(lambda i:text(m.mpos[0,i],m.mpos[1,i],str(i+1)),arange(m.num_mics))
show()