# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------

from distutils.core import setup, Extension
from os.path import join
import sys

import acoular
bf_version = str(acoular.__version__)
bf_author = str(acoular.__author__)

# preparation for compiling the beamformer extension
import scipy.weave
weavepath = scipy.weave.__path__[0]
import numpy
numpypath = numpy.__path__[0]

extra_compile_args = ['-O3','-ffast-math','-msse3', \
        '-Wno-write-strings','-fopenmp']
extra_link_args = ['-lgomp']

# we use OpenMP for Linux only
if sys.platform[:5] == 'linux':
    compiler = 'unix'
else:    
    extra_compile_args.pop()
    extra_link_args.pop()

module1 = Extension('acoular.beamformer',
                    define_macros = [('MAJOR_VERSION', '15'),
                                     ('MINOR_VERSION', '1.31')],
                    include_dirs = [weavepath, join(weavepath,'scxx'), 
                                    join(weavepath,'blitz'),
                                    join(numpypath,'core','include')],
                    compiler = compiler,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args,
                    sources = ['acoular/beamformer.cpp'])
    
setup(name="acoular", 
      version=bf_version, 
      description="Library for acoustic beamforming",
      author=bf_author,
      author_email="",
      url="www.acoular.org",
      packages = ['acoular'],
      ext_modules = [module1],
      scripts=['ResultExplorer.py','CalibHelper.py'],
      include_package_data = True,
      package_data={'acoular': ['xml/*.xml']}
)
#package_data={'acoular': ['doc/*.*','*.pyd','*.so','xml/*.xml']}