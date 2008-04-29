from beamfpy import *
from numpy import set_printoptions
from pylab import *
from time import time
set_printoptions(precision=2)
td=TimeSamples()
#~ td.name=".\\td\\24.01.2008 18_31_09,468.td"
td.name=".\\td\\15.11.2007 12_12_02,218.h5"

#~ td.configure_traits()
print td.data[0,0]
f=EigSpectra(window="Hanning",overlap="50%",block_size=1024)
f.time_data=td
m=MicGeom(from_file='acousticam_2c.xml')
#~ m=MicGeom(from_file='W90_D105_f10.xml')
g=RectGrid(x_min=-1,x_max=1,y_min=-1,y_max=1,z=0.68,increment=0.05)
#~ g=RectGrid(x_min=-0.6,x_max=0.4,y_min=-0.5,y_max=0.5,z=0.68,increment=0.025)
#~ g=RectGrid(x_min=-0.6,x_max=-0.1,y_min=-0.25,y_max=0.25,z=0.68,increment=0.03)
#~ g=RectGrid(x_min=-0.5,x_max=0.5,y_min=-0.5,y_max=0.5,z=1.0,increment=0.025)
#~ g=RectGrid(x_min=-0.6,x_max=0.6,y_min=-0.6,y_max=0.6,z=0.855,increment=0.05)
b=BeamformerBase(freq_data=f,grid=g,mpos=m)
#~ b=BeamformerEig(freq_data=f,grid=g,mpos=m,n=31)

fN=31
figure(1)
h1=b.result[fN]
print f.fftfreq()[fN]
L1=L_p(h1)
#~ subplot(2,2,1)
im=imshow(L1.swapaxes(0,1),vmin= max(ravel(L1))-10,vmax=max(ravel(L1)),origin='lower',extent=g.extend(),interpolation='bicubic')
colorbar()
xlim(g.extend()[0:2])
ylim(g.extend()[2:4])
ma=h1.max()/100
print len(h1.flatten()),sum(where(h1.flatten()>ma,1,0))


t=time()
#~ figure(2)
kj=array((2j*pi*f.fftfreq()[fN]/b.c,))
e = zeros((td.numchannels),'D')
h = zeros((g.size,g.size),'d')
gpos=g.pos()[:,:,newaxis]
mpos=m.mpos[:,newaxis,:]
rm=gpos-mpos
rm=sum(rm*rm,0)
rm=sqrt(rm)
p=exp(kj[0]*rm)/rm
print shape(p)
eva = ones((g.size,1),'d')
eve = p[:,:,newaxis]
print shape(eve)
print 'start',time()-t
#~ beamortho_sum(e,h,g.pos(),m.mpos,kj*ones(g.size,'d'),eva,eve,0,1)                             
hh = zeros((1,g.size,g.size),'d')
e1=e.copy()
beam_psf(e,e1,hh,g.pos(),m.mpos,kj)                             
h=hh[0]
print 'stop',time()-t
#~ h2=reshape(h,(g.size,g.nxsteps,g.nysteps))
#~ L2=L_p(h2[215]+h2[300])
#~ im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-30,vmax=max(ravel(L2)),origin='lower',extent=g.extend(),interpolation='bicubic')
#~ from scipy.linalg.iterative import gmres
figure(8)
p=h.copy()
p=p/diag(p)[:,newaxis]
imshow(p)
N=g.size
y=h1.flatten()
A=h/diag(h)[:,newaxis]-diag(ones(N,'d'))

j=1
err=[]
for om in arange(1.,2,1.4):
    x=zeros(N,'d')
    print time()-t
    #~ x=y
    gseidel(A,y,x,500,1.0)
    #~ for i in range(500):
       #~ for n in range(N):
            #~ x[n]=y[n]-dot(A[n],x)
            #~ if x[n]<0:
                #~ x[n]=0
        #~ err.append(x.nonzero()[0].size)
        #~ err.append(x.sum())
        #~ if i in (100,):
            #~ j+=1
            #~ subplot(2,2,j)
            #~ L2=L_p(x.reshape((g.nxsteps,g.nysteps)))
            #~ im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-30,vmax=max(ravel(L2)),origin='lower',extent=g.extend())#,interpolation='bicubic')
            #~ colorbar()
print time()-t
#~ x=y.copy()#zeros(N,'d')
#~ xi=set(arange(N))
#~ print time()-t
#~ for i in range(100):
    #~ for n in range(N):
        #~ xl=list(xi)
        #~ x[n]=y[n]-dot(A[n,xl],x[xl])
        #~ if x[n]<0:
            #~ x[n]=0
            #~ xi.discard(n)
        #~ else:
            #~ xi.add(n)
#~ print time()-t,xi

figure(2)
#~ x=dot(p,x)
L2=L_p(x.reshape((g.nxsteps,g.nysteps)))
im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-10,vmax=max(ravel(L2)),origin='lower',extent=g.extend())#,interpolation='bicubic')
#~ colorbar()
print x.sum(),sum(f.eva[fN])/32,sum(diag(f.csm[fN]))/32

#~ j=0
#~ figure(4)
#~ u=y.copy()

#~ for t in range(5000):
    #~ ci=dot(p,u)
    #~ u=u*dot(y,p)/ci
    #~ if t in (10,100,500,1000):
        #~ j+=1
        #~ subplot(2,2,j)
        #~ L2=L_p(u.reshape((g.nxsteps,g.nysteps)))
        #~ im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-30,vmax=max(ravel(L2)),origin='lower',extent=g.extend())#,interpolation='bicubic')
        #~ colorbar()

#~ figure(3)
#~ u=dot(p,u)
#~ L2=L_p(u.reshape((g.nxsteps,g.nysteps)))
#~ im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-10,vmax=max(ravel(L2)),origin='lower',extent=g.extend())#,interpolation='bicubic')
#~ colorbar()
#~ print u.sum()

#~ figure(3)
#~ e=abs(array(err)-sum(f.eva[fN])/32)/(sum(f.eva[fN])/32)
#~ semilogy(abs(e[:-1]-e[1:]))
#~ figure(4)
#~ semilogy(err)
#~ print time()-t
#~ AA=dot(p.T,p)*2
#~ Ay=dot(p.T,y)*2
#~ l=sum(f.eva[fN])/32
#~ x=linalg.solve(AA+diag(l*ones(N)/32),Ay)
#~ for i in range(10):
    #~ for n in range(N):
        #~ S=dot(AA[n],x)-AA[n,n]*x[n]-Ay[n]
        #~ if S>l:
            #~ x[n]=(l-S)/AA[n,n]
        #~ elif S<l:
            #~ x[n]=(-l-S)/AA[n,n]
        #~ else:
            #~ x[n]=0.0
#~ x=dot(p,x)
#~ L2=L_p(x.reshape((g.nxsteps,g.nysteps)))
#~ im=imshow(L2.swapaxes(0,1),vmin= max(ravel(L2))-10,vmax=max(ravel(L2)),origin='lower',extent=g.extend())#,interpolation='bicubic')
#~ colorbar()
#~ print x.sum(),sum(f.eva[fN])/32,sum(diag(f.csm[fN]))/32

show()
