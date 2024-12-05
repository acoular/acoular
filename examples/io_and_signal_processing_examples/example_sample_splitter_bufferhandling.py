# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Parallel processing chains -- SampleSplitter buffer handling.
=============================================================

This example shows the different behaviour of SampleSplitter class
when the maximum size of a block buffer is reached for one object obtaining
data.

Three different settings are available for the buffer overflow behaviour:
    * none: no warning, no error
    * warning: a warning appears
    * error: an error is raised
"""

import threading
from time import sleep

import acoular as ac
import numpy as np

# %%
# Set up data source. For convenience, we use a synthetic white noise with length of 1 s.

fs = 16000
ts = ac.TimeSamples(data=np.random.randn(fs * 1)[:, np.newaxis], sample_freq=fs)

# %%
# Connect SampleSplitter to data source

ss = ac.SampleSplitter(source=ts)


# %%
# Create two objects to process the time data. Registration of these objects will
# be done at the SampleSplitter object. For each object, a buffer will be created
# at the SampleSplitter object, which can store 5 blocks of data for each object.
# We set the buffer overflow behaviour to 'none' so that no warning or error is
# raised when the buffer is full. If the maximum size of the buffer is reached,
# the oldest block will be removed from the buffer.

tp1 = ac.TimePower(source=ss)
tp2 = ac.TimePower(source=ss)

# register these objects at SampleSplitter
ss.register_object(tp1, tp2, buffer_size=5, buffer_overflow_treatment='none')

# %%
# Define some useful functions for inspecting and for reading data from
# the SampleSplitter buffers. Three different functions are defined to
# simulate different processing speeds (fast, slow).


def print_number_of_blocks_in_block_buffers():
    """Prints the number of data blocks in SampleSplitter-buffers. For each
    subsequent object, a buffer exist.
    """
    buffers = list(ss.block_buffer.values())
    elements = [len(buf) for buf in buffers]
    print(f"num blocks in buffers: {dict(zip(['tp1','tp2'], elements))}")


def get_data_fast(obj):  # not time consuming function
    """Gets data fast (pause 0.01 seconds)"""
    for _ in obj.result(2048):  #
        print('tp1 calls sample splitter')
        print_number_of_blocks_in_block_buffers()
        sleep(0.01)


def get_data_slow(obj):  # more time consuming function
    """Gets data slow (pause 0.1 seconds)"""
    for i in obj.result(2048):  #
        print('tp2 calls sample splitter')
        print_number_of_blocks_in_block_buffers()
        sleep(0.1)


# %%
# Prepare and start processing in threads
# (no warning or error when block buffer is full)

print("buffer overflow behaviour == 'none'")

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print('start threads')

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print('threads finished')


# %%
# Prepare and start processing in threads
# (only warning when block buffer is full)

print("buffer overflow behaviour == 'warning'")

ss.buffer_overflow_treatment[tp1] = 'warning'
ss.buffer_overflow_treatment[tp2] = 'warning'

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print('start threads')

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print('threads finished')

# %%
# Prepare and start processing in threads
# (raise error when block buffer is full)

print("buffer overflow behaviour == 'error'")

ss.buffer_overflow_treatment[tp1] = 'error'
ss.buffer_overflow_treatment[tp2] = 'error'

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print('start threads')

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print('threads finished')

# %%
