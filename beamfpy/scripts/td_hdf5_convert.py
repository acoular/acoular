# Umwandlung eines .td-files in ein hdf5-File

from beamfpy import *
from os import path, walk

t=TimeSamples()
ti=td_import()
ti.configure_traits(kind='modal')
d,f=path.split(ti.from_file)
for dp,dn,fn in walk(d):
    print dp
    if dp==d:
        for name in fn:
            r,e = path.splitext(name)
            if e=='.td':
                tn=path.join(dp,name)
                print tn
                ti.from_file=tn
                t.name=""
                ti.get_data(t)
                t.h5f.close()
                print t.h5f

                
