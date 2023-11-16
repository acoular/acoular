
from os import path
from numpy import array, round
import matplotlib.pyplot as plt
from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource,\
    Mixer, WriteH5, TimeSamples, PowerSpectra, RectGrid, SteeringVector,\
    BeamformerCleansc, L_p, ImportGrid,CircSector
from acoular.tools import MetricEvaluator

# set up the parameters
sfreq = 51200 
duration = 1
nsamples = duration*sfreq
micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
h5savefile = 'three_sources.h5'

# generate test data, in real life this would come from an array measurement
mg = MicGeom( from_file=micgeofile )
n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
n2 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7 )
n3 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5 )
p1 = PointSource( signal=n1, mics=mg,  loc=(-0.1,-0.1,0.3) )
p2 = PointSource( signal=n2, mics=mg,  loc=(0.15,0,0.3) )
p3 = PointSource( signal=n3, mics=mg,  loc=(0,0.1,0.3) )
pa = Mixer( source=p1, sources=[p2,p3] )
wh5 = WriteH5( source=pa, name=h5savefile )
wh5.save()

# analyze the data and generate map

ts = TimeSamples( name=h5savefile )
ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning' )

rg = RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
increment=0.01 )
st = SteeringVector(grid=rg, mics=mg, ref = 1.)

bb = BeamformerCleansc( freq_data=ps, steer=st )
pm = bb.synthetic( 8000, 0 )
Lm = L_p( pm )

# evaluate the results 
target_grid = ImportGrid(
    gpos_file= array([
    list(p1.loc),
    list(p2.loc),
    list(p3.loc)
]).T)

nfft = ps.fftfreq().shape[0]

target_data = array([
    [n1.rms**2/nfft],
    [n2.rms**2/nfft],
    [n3.rms**2/nfft]]).T

mv = MetricEvaluator(
    sector = CircSector(r=0.05*mg.aperture),
    grid = rg,
    data = pm.reshape((1,-1)),
    target_grid = target_grid,
    target_data = target_data,
)

# plot the data

plt.figure()
# show map
plt.imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
interpolation='none')
# plot sectors
ax = plt.gca()
for j, sector in enumerate(mv.sectors):
    ax.add_patch(plt.Circle((sector.x, sector.y), 
                sector.r, color="black",fill=False))
    # annotate specific level error below circles
    plt.annotate(r'$\Delta L_{p,e,s}$='+ str(round(mv.get_specific_level_error()[0,j],2)) + ' dB',
                xy=(sector.x-0.1,sector.y-sector.r-0.01), color='white')
# annotate overall level error
plt.annotate(r'$\Delta L_{p,e,o}$='+ str(round(mv.get_overall_level_error()[0],2)) + ' dB',
            xy=(0.05,0.95),xycoords='axes fraction', color='white')
plt.annotate(r'$\Delta L_{p,e,i}$='+ str(round(mv.get_inverse_level_error()[0],2)) + ' dB',
            xy=(0.6,0.95),xycoords='axes fraction', color='white')    
plt.colorbar()
