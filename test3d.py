import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, EigSpectra,\
RectGrid3D, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc

from os import path



t = TimeSamples(name=path.join(td_dir,'2008-05-16_11-36-00_468000.h5'))

# Kalibrierdaten
cal = Calib(from_file=path.join(td_dir,'calib_06_05_2008.xml'))

# Mikrofongeometrie
m = MicGeom(from_file=path.join( path.split(beamfpy.__file__)[0],'xml','array_56.xml'))

f = EigSpectra(window='Hanning',overlap='50%',block_size=128,ind_low=5,ind_high=15)
f.time_data = t
print f.fftfreq()[10]
g = RectGrid3D(x_min=-0.6,x_max=-0.0,y_min=-0.3,y_max=0.3,z_min=0.48,z_max=0.88,\
increment=0.05)

b = BeamformerBase(freq_data=f,grid=g,mpos=m,r_diag=True,c=346.04)
#bc = BeamformerCapon(freq_data=f,grid=g,mpos=m,c=346)
be = BeamformerEig(freq_data=f,grid=g,mpos=m,r_diag=True,c=346,n=50)
#bm = BeamformerMusic(freq_data=f,grid=g,mpos=m,r_diag=True,c=346)
#bd = BeamformerDamas(beamformer=b,n_iter=100)
bo = BeamformerOrth(beamformer=be, eva_list=range(40,56))
bs = BeamformerCleansc(freq_data=f,grid=g,mpos=m,r_diag=True,c=346)
map = b.synthetic(4000,1)
print map.shape
L1 = L_p(map)
mx = L1.max()
print(L_p(b.integrate((-0.3,-0.1,0.58,-0.1,0.1,0.78)))[f.ind_low:f.ind_high])

from numpy import mgrid, arange
from enthought.mayavi import mlab
X,Y,Z = mgrid[g.x_min:g.x_max:1j*g.nxsteps,\
            g.y_min:g.y_max:1j*g.nysteps,\
            g.z_min:g.z_max:1j*g.nzsteps]
print X.shape
#~ mlab.contour3d(X,Y,Z,L1,vmin=mx-5,vmax=mx,transparent=True)#,contours=[49.0,50.0,51.0,52.0,53.0])
#~ mlab.points3d(X,Y,Z,L1,transparent=True)#,contours=[49.0,50.0,51.0,52.0,53.0])
data = mlab.pipeline.scalar_field(X,Y,Z,L1)
mlab.pipeline.iso_surface(data,contours=arange(mx-10,mx,1).tolist(),vmin=mx-10,vmax=mx)
s = L1
#~ mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=0, vmax=0.1)

mlab.axes()
#~ mlab.show_pipeline()
mlab.show()


