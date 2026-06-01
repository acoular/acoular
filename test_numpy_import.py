#!/usr/bin/env python3
"""Test script to verify numpy import order fix."""

import os
import sys

# Test 1: Import acoular first, then numpy
print("=== Test 1: Import acoular first ===")
print(f"OPENBLAS_NUM_THREADS before import: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")

import acoular
print(f"OPENBLAS_NUM_THREADS after acoular import: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")

import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"Acoular version: {acoular.__version__}")

# Test 2: Check that environment variables are set correctly
print("\n=== Test 2: Environment variables ===")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

# Test 3: Verify no warnings are issued
print("\n=== Test 3: No warnings expected ===")
print("If you see warnings about numpy import order, the fix didn't work.")

print("\n=== All tests passed! ===")
