from beamfpy import *
from numpy import *
from pylab import *
from beamfpy.beamfpy import td_dir, PointSpreadFunction
from beamfpy.beamformer import *
from os import path
set_printoptions(precision=2)
from time import time

print beamfpy.__file__

m=MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))
g=RectGrid(x_min=-0.5,x_max=0.5,y_min=-0.5,y_max=0.5,z=0.68,increment=0.05)
freqs=array((10000.0,))
#~ p=PointSpreadFunction(freqs=freqs,mpos=m,grid=g)
#~ print p.psf().shape

numfreq = 1
kj = 2j*pi*freqs/343
gs = g.size
print gs
bpos = g.pos()
mpos=m.mpos
hh = ones((numfreq,gs,gs),'d')
print hh.size
e = zeros((56),'D')
e1 = e.copy()
t=time()
beam_psf(e,e1,hh,bpos,m.mpos,kj)
print "psf:",time()-t
for i in range(numfreq):
    h = hh[i]
    hh[i] = h/diag(h)[:,newaxis]
#~ print time()-t

psf0=hh[0,gs/2].reshape(g.nxsteps,g.nysteps)
imshow(L_p(psf0)-94,vmin=-20)
colorbar()
x=ones(gs,'d')
y=ones(gs,'d')
t=time()
gseidel(hh[0],x,y,500,1.0)
print "gseidel",time()-t


figure(2)
#~ mpos=mpos.swapaxes(0,1)
#~ bpos=bpos.swapaxes(0,1)
#~ print shape(bpos),shape(mpos)
#~ print bpos[:,0],mpos[:,0]
rm=bpos[:,:,newaxis]-mpos[:,newaxis,:]
rm=sum(rm*rm,0)
rm=sqrt(rm)
#~ print shape(rm),rm[0]
r0=sum(bpos*bpos,0)
r0=sqrt(r0)
#~ print shape(r0),r0[0]

e=array(e,'F')
e1=array(e1,'F')
hh=array(hh,'f')
rm=array(rm,'f')
r0=array(r0,'f')
kj=array(kj,'F')


t=time()
beam_psf1(e,e1,hh,rm,r0,kj)
print "psf1",time()-t
for i in range(numfreq):
    h = hh[i]
    hh[i] = h/diag(h)[:,newaxis]
#~ print time()-t

psf0=hh[0,gs/2].reshape(g.nxsteps,g.nysteps)
imshow(L_p(psf0)-94,vmin=-20)
colorbar()



x=ones(gs,'f')
y=ones(gs,'f')
hh=array(hh,'f')
t=time()
gseidel1(hh[0],x,y,500)
print "gseidel1",time()-t


show()