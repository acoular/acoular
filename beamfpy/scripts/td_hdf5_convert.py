# Umwandlung eines .td-files in ein hdf5-File

import numpy
import time
import tables
import cPickle

f=file('.\\td\\07.03.2007 16_45_40,671.td','rb')
h=cPickle.load(f)
f.close()
sample_freq = h['sample_freq']
data = h['data']
(numsamples,numchannels)=data.shape


f5h=tables.openFile('test.h5',mode='w')
#group=f5h.createGroup(f5h.root,'time_data')
ac=f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,numchannels))
ac.setAttr('sample_freq',sample_freq)
ac.append(data)
f5h.close()

f5h=tables.openFile('test.h5')
ac=f5h.root.time_data
print ac.getAttr('sample_freq'),ac.shape
    
from pylab import *
plot(ac[:1000,0])
    
plot(ac[:1000,7])

show()
f5h.close()
#~ print "End of program, press Enter key to quit"
#~ raw_input()