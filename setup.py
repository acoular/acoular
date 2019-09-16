# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------


from setuptools import setup
from os.path import join, abspath, dirname
import os


#import acoular.version as av
#bf_version = str(av.__version__)
#bf_author = str(av.__author__)

bf_version = "19.08"
bf_author = "Acoular Development Team"


# Get the long description from the relevant file
here = abspath(dirname(__file__))
with open(join(here, 'README.rst')) as f:
    long_description = f.read()


install_requires = list([
      'numpy>=1.11.3',
      'setuptools',	
      'numba >=0.40.0',
      'scipy<=0.12.0;python_version<="2.7"',
      'scipy>=0.1.0;python_version>="3.4"',
      'scikit-learn<=0.20.0;python_version<="2.7"',
      'scikit-learn>=0.19.1;python_version>="3.4"',
      'tables>=3.4.4',
      'traits>=4.6.0',
      #'traitsui>=6.0.0',
      'configparser;python_version<"3.4"',
	]),

if "CONDA_PREFIX" not in os.environ:
    install_requires.append('python-qt5;python_version<="2.7"',
                            'PyQt5>=5.6;python_version>="3.4"',)

setup_requires = list([
      'numpy>=1.11.3',
      'setuptools',	
      #'python-qt5;python_version<="2.7"',
      #'PyQt5>=5.6;python_version>="3.4"',
      'numba >=0.40.0',
      'scipy<=0.12.0;python_version<="2.7"',
      'scipy>=0.1.0;python_version>="3.4"',
      'scikit-learn<=0.20.0;python_version<="2.7"',
      'scikit-learn>=0.19.1;python_version>="3.4"',
      'tables>=3.4.4',
      'traits>=4.6.0',
      #'traitsui>=6.0.0',
      'configparser;python_version<="2.7"',
      #'libpython; platform_system == "Windows"',
	]),

if "CONDA_PREFIX" not in os.environ:
    setup_requires.append('python-qt5;python_version<="2.7"',
                          'PyQt5>=5.6;python_version>="3.4"',)

    
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
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      ],
      keywords='acoustic beamforming microphone array',
      packages = ['acoular'],

      install_requires = install_requires,

      setup_requires = setup_requires,
      
      scripts=[join('examples','acoular_demo.py')],
      include_package_data = True,
      package_data={'acoular': ['xml/*.xml'],
		    'acoular': ['tests/*.*']},
      #to solve numba compiler 
      zip_safe=False
)

