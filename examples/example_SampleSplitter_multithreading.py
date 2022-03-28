# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""
This Examples shows how to use the SampleSplitter class in a multithreading 
(3 threads) scenario.

Three different objects (tp1, tp2 and tp3) obtain and process time data 
from the same MaskedTimeSamples object in parallel. The SampleSplitter class
is used to split the data stream for parallel processing.
"""

from acoular import TimePower,MaskedTimeSamples, SampleSplitter
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
# create three objects to process the data
# =============================================================================

tp1 = TimePower(source=ss)
tp2 = TimePower(source=ss)
tp3 = TimePower(source=ss)

# register these objects at SampleSplitter
ss.register_object(tp1,tp2,tp3) # register objects

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
    """ gets data fast (pause 0.3 seconds)"""
    for _ in obj.result(2048): # 
        print("tp1 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.3) 

def get_data_mid(obj): # not time consuming function
    """ gets data mid speed (pause 0.5 seconds)"""
    for i in obj.result(2048): # 
        print("tp2 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.5) 

def get_data_slow(obj): # more time consuming function
    """ gets data slow (pause 0.7 seconds)"""
    for i in obj.result(2048): # 
        print("tp3 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.7) 

# =============================================================================
# prepare and start processing in threads
# =============================================================================

worker1 = threading.Thread(target=get_data_fast, args=(tp1,))
worker2 = threading.Thread(target=get_data_mid, args=(tp2,))
worker3 = threading.Thread(target=get_data_slow, args=(tp3,))

print("start threads")

worker1.start()
worker2.start()
worker3.start()

[thr.join() for thr in [worker1,worker2,worker3]]

print("threads finished")
