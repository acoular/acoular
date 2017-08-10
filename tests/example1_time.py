#===============================================================================
# acoular example1.py without frequency beamformers
#===============================================================================



from os import path
import acoular
from pylab import figure, imshow, colorbar, show, subplot,title
from numpy import zeros


micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_56.xml')
datafile = 'example_data.h5'
calibfile = 'example_calib.xml'
cfreq = 4000

mg = acoular.MicGeom( from_file=micgeofile )
ts = acoular.MaskedTimeSamples( name=datafile )

ts.start = 0
ts.stop = 16000
invalid = [1,7]
ts.invalid_channels = invalid
mg.invalid_channels = invalid
rg = acoular.RectGrid( x_min=-0.6, x_max=0.0, y_min=-0.3, y_max=0.3, z=0.68, \
increment=0.05 )
bt = acoular.BeamformerTime(source = ts, grid = rg, mpos = mg, c=346.04)
ft = acoular.FiltFiltOctave(source=bt, band=cfreq)
pt = acoular.TimePower(source=ft)
avgt = acoular.TimeAverage(source=pt, naverage = 1024)
cacht = acoular.TimeCache( source = avgt) # cache to prevent recalculation

#===============================================================================
# delay and sum beamformer in time domain with autocorrelation removal
# processing chain: zero-phase filtering, beamforming+power, average
#===============================================================================

fi = acoular.FiltFiltOctave(source=ts, band=cfreq)
bts = acoular.BeamformerTimeSq(source = fi,grid=rg, mpos=mg, r_diag=True,c=346.04)
avgts = acoular.TimeAverage(source=bts, naverage = 1024)
cachts = acoular.TimeCache( source = avgts) # cache to prevent recalculation

#===============================================================================
# plot result maps for different beamformers in time domain
#===============================================================================
i1 = 1
i2 = 2 # no of figure
for block in (cacht, cachts): 
    # first, plot time-dependent result (block-wise)
    figure(i2)
    i2 += 1
    res = zeros(rg.size) # init accumulator for average
    i3 = 1 # no of subplot
    for block_res in block.result(1):  #one single block
        subplot(4,4,i3)
        i3 += 1
        res += block_res[0] # average accum.
        map = block_res[0].reshape(rg.shape)
        mx = acoular.L_p(map.max())
        imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, 
               interpolation='nearest', extent=rg.extend())
        title('%i' % ((i3-1)*1024))
    res /= i3-1 # average
    # second, plot overall result (average over all blocks)
    figure(1)
    subplot(3,4,i1)
    i1 += 1
    map = res.reshape(rg.shape)
    mx = acoular.L_p(map.max())
    imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=rg.extend())
    colorbar()
    title(('BeamformerTime','BeamformerTimeSq')[i2-3])
show()    

