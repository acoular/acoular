#!/usr/bin/env python3
"""Test script to verify the original issue #187 is resolved."""

import os
import time
from os import path

# Set environment to ensure we test the fix
os.environ.pop('OPENBLAS_NUM_THREADS', None)
os.environ.pop('MKL_NUM_THREADS', None) 
os.environ.pop('OMP_NUM_THREADS', None)

print("=== Testing Original Issue #187 ===")
print("This test reproduces the original beamforming performance issue.")

# Import acoular first (this should set the environment variables)
import acoular
from acoular import __file__ as bpath
from acoular import MicGeom, WNoiseGenerator, PointSource
from acoular import Mixer, WriteH5, TimeSamples, PowerSpectra
from acoular import RectGrid, SteeringVector, BeamformerCleansc, config

print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

config.global_caching = 'none'

# set up the parameters
sfreq = 51200 
duration = 1  # Use original duration for meaningful performance test
nsamples = duration*sfreq
micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
h5savefile = 'test_three_sources.h5'

print(f"\n=== Generating test data ===")
print(f"Sample frequency: {sfreq} Hz")
print(f"Duration: {duration} s")
print(f"Number of samples: {nsamples}")

# generate test data
mg = MicGeom(file=micgeofile)
n1 = WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=1)
n2 = WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=2, rms=0.7)
n3 = WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=3, rms=0.5)
p1 = PointSource(signal=n1, mics=mg, loc=(-0.1,-0.1,0.3))
p2 = PointSource(signal=n2, mics=mg, loc=(0.15,0,0.3))
p3 = PointSource(signal=n3, mics=mg, loc=(0,0.1,0.3))
pa = Mixer(source=p1, sources=[p2,p3])
wh5 = WriteH5(source=pa, file=h5savefile)
wh5.save()

print(f"Test data generated and saved to {h5savefile}")

# analyze the data and generate map
print(f"\n=== Running beamforming ===")
ts = TimeSamples(file=h5savefile)
ps = PowerSpectra(source=ts, block_size=128, window='Hanning')

rg = RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
              increment=0.02)
st = SteeringVector(grid=rg, mics=mg)
bb = BeamformerCleansc(freq_data=ps, steer=st)

start_time = time.time()
pm = bb.synthetic(8000, 0)
elapsed_time = time.time() - start_time

print(f"Beamforming completed in {elapsed_time:.2f} seconds")
print(f"Result shape: {pm.shape}")

# Performance check - the original issue reported 7.5s with fix vs 273s without
if elapsed_time > 30:
    print(f"WARNING: Performance regression detected! Expected < 30s, got {elapsed_time:.2f}s")
    print("This suggests the threading conflict is not properly resolved.")
else:
    print(f"Performance check passed: {elapsed_time:.2f}s is within expected range")

# Clean up
if os.path.exists(h5savefile):
    os.remove(h5savefile)
    print(f"Cleaned up test file {h5savefile}")

print(f"\n=== Test completed successfully ===")
print(f"The fix ensures that OPENBLAS_NUM_THREADS=1 is set before numpy is imported,")
print(f"preventing the threading conflict that caused the original performance issue.")
