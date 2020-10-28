# -*- coding: utf-8 -*-
"""
Plot Acoular examples

Runs the example scripts found in the current directoy (example*.py)
and exports generated figures.
This is for easily creating the figures used for the documentation.

Usage: 
- Copy script into acoular/examples/ directory
- Run in terminal
- Move generated *.png files to acoular/docs/source/examples/

Do not run this script in an ipython console.

Copyright (c) 2018 The Acoular developers.
All rights reserved.
"""

from glob import glob
from importlib import import_module
from pylab import get_fignums, figure, savefig, close

for ex in glob("example*.py"):
    exname = ex[:-3]
    print('Importing %s ...' % exname)
    try:
        import_module(exname)
        for fn in get_fignums():
            figure(fn)
            figname = exname + '_' + str(fn)+ '.png'
            print('Exporting %s ...' % figname)
            savefig(figname, bbox_inches='tight')
        close('all')
    except:
        print('---------------------------------------------')
        print('        Error importing %s !' % ex)
        print('---------------------------------------------')
