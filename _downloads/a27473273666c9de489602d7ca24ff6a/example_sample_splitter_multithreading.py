# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Parallel processing chains -- Multithreading with the SampleSplitter.
=====================================================================

This Examples shows how to use the SampleSplitter class in a multithreading
(3 threads) scenario.

Three different objects (tp1, tp2 and tp3) obtain and process time data
from the same MaskedTimeSamples object in parallel. The SampleSplitter class
is used to split the data stream for parallel processing.
"""

import threading
from time import sleep

import acoular as ac
import numpy as np

# %%
# Set up data source. For convenience, we use a synthetic white noise with length of 1 s.

fs = 8192
ts = ac.TimeSamples(data=np.random.randn(fs * 1)[:, np.newaxis], sample_freq=fs)

# %%
# Connect SampleSplitter to data source

ss = ac.SampleSplitter(source=ts)


# %%
# Create three objects to process the data

tp1 = ac.TimePower(source=ss)
tp2 = ac.TimePower(source=ss)
tp3 = ac.TimePower(source=ss)

# register these objects at SampleSplitter
ss.register_object(tp1, tp2, tp3)  # register objects

# %%
# Define some useful functions for inspecting and for reading data from
# the SampleSplitter buffers. Three different functions are defined to
# simulate different processing speeds (fast, mid, slow).


def print_number_of_blocks_in_block_buffers():
    """Prints the number of data blocks in SampleSplitter-buffers. For each
    subsequent object, a buffer exist.
    """
    buffers = list(ss.block_buffer.values())
    elements = [len(buf) for buf in buffers]
    print(f"num blocks in buffers: {dict(zip(['tp1','tp2','tp3'], elements))}")


def get_data_fast(obj):  # not time consuming function
    """Gets data fast (pause 0.03 seconds)"""
    for _ in obj.result(2048):  #
        print('tp1 calls sample splitter')
        print_number_of_blocks_in_block_buffers()
        sleep(0.03)


def get_data_mid(obj):  # not time consuming function
    """Gets data mid speed (pause 0.05 seconds)"""
    for i in obj.result(2048):  #
        print('tp2 calls sample splitter')
        print_number_of_blocks_in_block_buffers()
        sleep(0.05)


def get_data_slow(obj):  # more time consuming function
    """Gets data slow (pause 0.07 seconds)"""
    for i in obj.result(2048):  #
        print('tp3 calls sample splitter')
        print_number_of_blocks_in_block_buffers()
        sleep(0.07)


# %%
# Prepare and start processing in threads

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_mid, args=(tp2,))
worker3 = threading.Thread(target=get_data_slow, args=(tp3,))

print('start threads')

worker1.start()
worker2.start()
worker3.start()

[thr.join() for thr in [worker1, worker2, worker3]]

print('threads finished')
