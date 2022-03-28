# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""
This example shows the different behaviour of SampleSplitter class  
when the maximum size of a block buffer is reached for one object obtaining 
data.

Three different settings can be made by the user:
    * none: no warning, no error
    * warning: a warning appears
    * error: an error is raised
"""

from acoular import TimePower, MaskedTimeSamples, SampleSplitter
import threading
from time import sleep

samples = 25000

# =============================================================================
#  set up data source
# =============================================================================
h5savefile = 'example_data.h5'
ts = MaskedTimeSamples(name=h5savefile,
                       start = 0,
                       stop = samples)  

# =============================================================================
# connect SampleSplitter to data source
# =============================================================================

# set up Sample Splitter
ss = SampleSplitter(source = ts)


# =============================================================================
# create two objects to process the data
# =============================================================================

tp1 = TimePower(source=ss)
tp2 = TimePower(source=ss)

# register these objects at SampleSplitter
ss.register_object(tp1,tp2) # register objects

# =============================================================================
# define functions 
# =============================================================================

def print_number_of_blocks_in_block_buffers():
    """ 
    prints the number of data blocks in SampleSplitter-buffers. For each
    subsequent object, a buffer exist.
    """
    buffers = list(ss.block_buffer.values())
    elements = [len(buf) for buf in buffers]
    print(f"num blocks in buffers: {dict(zip(['tp1','tp2','tp3'], elements))}")


def get_data_fast(obj): # not time consuming function
    """ gets data fast (pause 0.1 seconds)"""
    for _ in obj.result(2048): # 
        print("tp1 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.1) 

def get_data_slow(obj): # more time consuming function
    """ gets data slow (pause 0.8 seconds)"""
    for i in obj.result(2048): # 
        print("tp3 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.8) 


# =============================================================================
# prepare and start processing in threads
# (no warning or error when block buffer is full)
# =============================================================================


print("buffer overflow behaviour == 'none'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'none'
ss.buffer_overflow_treatment[tp2] = 'none'

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")


# =============================================================================
# prepare and start processing in threads
# (only warning when block buffer is full)
# =============================================================================

print("buffer overflow behaviour == 'warning'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'warning'
ss.buffer_overflow_treatment[tp2] = 'warning'

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")

# =============================================================================
# prepare and start processing in threads
# (raise error when block buffer is full)
# =============================================================================

print("buffer overflow behaviour == 'error'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'error'
ss.buffer_overflow_treatment[tp2] = 'error'

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_slow, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")