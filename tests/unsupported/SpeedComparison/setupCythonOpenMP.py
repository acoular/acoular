#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Setup- Datei fuer Paralleles Cython (benutzt OpemMP).
Siehe "http://cython.readthedocs.io/en/latest/src/userguide/parallelism.html"
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#==============================================================================
## Laut Cython Doku "http://cython.readthedocs.io/en/latest/src/userguide/parallelism.html#Compiling" muss dieser Code noch rein
ext_modules = [
      Extension(
          "cythonBeamformer",
          ["cythonBeamformer.pyx"],
          extra_compile_args=['-fopenmp'],
          extra_link_args=['-fopenmp'],
      )
 ]
setup(
     name='cythonBeamformer',
     ext_modules=cythonize(ext_modules),
 )
#==============================================================================


#==============================================================================
## Compiler optionen von Acoular
#ext_modules = [
#       Extension(
#           "cythonBeamformer",
#           ["cythonBeamformer.pyx"],
#           extra_compile_args=['-O3','-ffast-math','-msse3', \
#        '-Wno-write-strings', '-fopenmp'],
#           extra_link_args=['-lgomp'],
#       )
#  ]
#setup(
#      name='cythonBeamformer',
#      ext_modules=cythonize(ext_modules),
#  )
#==============================================================================


#==============================================================================
## Compiler optionen nach "http://nealhughes.net/parallelcomp2/"
# ext_modules=[
#     Extension("cythonBeamformer",
#               ["cythonBeamformer.pyx"],
#               libraries=["m"],
#               extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#               extra_link_args=['-fopenmp']
#               ) 
# ]
# 
# setup( 
#   name = "cythonBeamformer",
#   cmdclass = {"build_ext": build_ext},
#   ext_modules = ext_modules
# )
#==============================================================================
