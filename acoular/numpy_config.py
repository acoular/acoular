# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Configure numpy/OpenBLAS environment before numpy is imported.

This module must be imported before any other module that might import numpy.
It sets up the environment to prevent threading conflicts between OpenBLAS and Numba.
"""

import os
import sys
from os import environ


def configure_numpy_environment():
    """Configure environment variables for numpy/OpenBLAS before numpy is imported.
    
    This function should be called as early as possible in the import process.
    It sets OPENBLAS_NUM_THREADS=1 to prevent threading conflicts with Numba.
    """
    # Set OPENBLAS_NUM_THREADS=1 to prevent overcommittment with Numba
    # This must be set before numpy is imported to have any effect
    if environ.get('OPENBLAS_NUM_THREADS') is None:
        environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Also set MKL_NUM_THREADS=1 for consistency with MKL builds
    if environ.get('MKL_NUM_THREADS') is None:
        environ['MKL_NUM_THREADS'] = '1'
    
    # Set OMP_NUM_THREADS=1 for OpenMP builds
    if environ.get('OMP_NUM_THREADS') is None:
        environ['OMP_NUM_THREADS'] = '1'


# Configure environment immediately when this module is imported
configure_numpy_environment()
