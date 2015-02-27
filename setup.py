# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2015, Acoular Development Team.
#------------------------------------------------------------------------------

from setuptools import setup, Extension
from os.path import join, abspath, dirname
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
#                    compiler = compiler,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args,
                    sources = ['acoular/beamformer.cpp'])
                    
# Get the long description from the relevant file
here = abspath(dirname(__file__))
with open(join(here, 'README.rst')) as f:
    long_description = f.read()
                    
    
setup(name="acoular", 
      version=bf_version, 
      description="Library for acoustic beamforming",
      long_description=long_description,
      license="BSD",
      author=bf_author,
      author_email="info@acoular.org",
      url="www.acoular.org",
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Physics',
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 2.7',
      ],
      keywords='acoustic beamforming microphone array',
      packages = ['acoular'],
      install_requires = [
      'numpy>=1.8',
      'scipy>=0.13',
      'sklearn>=0.15',
      'tables>=3.1',
      'traits>=4'],
      ext_modules = [module1],
      scripts=['ResultExplorer.py','CalibHelper.py'],
      include_package_data = True,
      package_data={'acoular': ['xml/*.xml']}
)
#package_data={'acoular': ['doc/*.*','*.pyd','*.so','xml/*.xml']}