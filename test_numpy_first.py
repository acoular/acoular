#!/usr/bin/env python3
"""Test script to verify numpy import order fix when numpy is imported first."""

import os
import sys

# Test: Import numpy first, then acoular
print("=== Test: Import numpy first ===")
print(f"OPENBLAS_NUM_THREADS before numpy import: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")

import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"OPENBLAS_NUM_THREADS after numpy import: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")

import acoular
print(f"Acoular version: {acoular.__version__}")
print(f"OPENBLAS_NUM_THREADS after acoular import: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")

# Check that environment variables are set correctly
print("\n=== Environment variables ===")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

print("\n=== Test completed ===")
print("Note: In this case, numpy was already imported, so the environment variables")
print("were set by acoular's configuration, but numpy's threading behavior is already determined.")
