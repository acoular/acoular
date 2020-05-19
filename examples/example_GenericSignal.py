# -*- coding: utf-8 -*-
"""

"""
from pylab import *
from acoular import *




# files
datafile = 'example_data.h5'

t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
invalid = [1,7] # list of invalid channels (unwanted microphones etc.)
t1.invalid_channels = invalid 

t2 = ChannelMixer(source=t1)

sig = GenericSignalGenerator(source=t2)

plot(sig.signal())
show()