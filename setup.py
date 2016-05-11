# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2016, Acoular Development Team.
#------------------------------------------------------------------------------

from setuptools import setup, Extension
from os.path import join, abspath, dirname
from os import remove
from shutil import copy
import sys

#import acoular.version as av
#bf_version = str(av.__version__)
#bf_author = str(av.__author__)

bf_version = "16.5"
bf_author = "Acoular developers"


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

# we need to link libgcc and libstdc++ statically for Windows
if sys.platform[:3] == 'win':
    extra_link_args.append('-static-libgcc')    
    extra_link_args.append('-static-libstdc++')

# provide weave_imp.cpp
copy(join(weavepath,'scxx','weave_imp.cpp'),'weave_imp.cpp')

# build C++ extension from weave-generated beamformer.cpp file
module1 = Extension('acoular.beamformer',
                    define_macros = [('MAJOR_VERSION', '16'),
                                     ('MINOR_VERSION', '5')],
                    include_dirs = [weavepath, join(weavepath,'scxx'), 
                                    join(weavepath,'blitz'),
                                    join(numpypath,'core','include')],
                    language = "c++",
#                    compiler = compiler,
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args,
                    sources = [
                        join('acoular','beamformer.cpp'),
                        'weave_imp.cpp'
                    ])

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
      url="http://www.acoular.org",
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
#      install_requires = [
#      'numpy>=1.8',
#      'scipy>=0.13',
#      'scikit-learn>=0.15',
#      'tables>=3.1',
#      'traits>=4.4',
#      'chaco>=4.4'],
#      setup_requires = [
#      'numpy>=1.8',
#      'scipy>=0.13',
#      'scikit-learn>=0.15',
#      'tables>=3.1',
#      'traits>=4.4',
#      'chaco>=4.4'],
      ext_modules = [module1],
      scripts=['ResultExplorer.py','CalibHelper.py',
               join('examples','acoular_demo.py')],
      include_package_data = True,
      package_data={'acoular': ['xml/*.xml']}
)

# cleanup
remove('weave_imp.cpp')