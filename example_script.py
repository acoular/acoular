"""
Script for the manipulation of the ResultExplorer object rx
rx Traits:

# beamformer, allows to change settings for TimeSamples and Spectra objects as well
beamformer = Trait(BeamformerBase, desc = 'beamformer object')

# bmap
bmap = Any(desc = 'actual beamforming map')

# dynamic_range
dynamic_range = Float(10.0, desc = 'dynamic range in decibel')

# map_interp
map_interp = Trait('bicubic', ('nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'), desc = 'method of interpolation applied to map')

# max_level
max_level = Trait('auto', 'auto', Float, desc = 'maximum level displayed')

# mplw
mplw = Instance(MPLWidget)

# pic_flag
pic_flag = Bool(False, desc = 'show picture ?')

# pic_scale
pic_scale = Float(400, desc = 'maximum  x-value picture plane')

# pic_x_min
pic_x_min = Float(-Const(1.0), desc = 'minimum  x-value picture plane')

# pic_y_min
pic_y_min = Float(-Const(0.75), desc = 'minimum  y-value picture plane')

# pict
pict = File(filter = [ '*.png' ], desc = 'name of picture file')

# synth_freq
synth_freq = Float(desc = 'mid frequency for band synthesis')

# synth_freq_enum
synth_freq_enum = List

# synth_type
synth_type = Trait('Oktave', { Const('Single frequency') : Const(0), Const('Oktave') : Const(1), Const('Third octave') : Const(3) }, desc = 'type of frequency band for synthesis')
"""
print script
rx.beamformer.grid = RectGrid(x_min=-1.5,x_max=3.0,y_min=0.0,y_max=2.1,z=5.0,increment=0.1)
rx.beamformer.mpos.from_file = path.join( path.split(beamfpy.__file__)[0],'xml','HW90D240_f10.xml')
rx.beamformer.freq_data.time_data.name = path.join(td_dir,'2_15_3mm_VA.h5')
rx.beamformer.freq_data.ind_low = 10
rx.beamformer.freq_data.ind_high = 12
rx.dynamic_range=10.0