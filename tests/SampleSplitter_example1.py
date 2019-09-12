# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------
"""
This Examples shows the use of SampleSplitter class in a multithreading scenario
"""

from acoular import TimePower,MaskedTimeSamples, SampleSplitter
import threading
from time import sleep
import numpy as np

samples = 25000

h5savefile = 'example_data.h5'
ts = MaskedTimeSamples(name=h5savefile,
                       start = 0,
                       stop = samples)  

# set up Sample Splitter
ss = SampleSplitter(source = ts)

# set up following objects
tp1 = TimePower(source=ss)
tp2 = TimePower(source=ss)
tp3 = TimePower(source=ss)

ss.register_object(tp1,tp2,tp3) # register objects

init_array1 = np.empty((samples, ts.numchannels),dtype=np.float32)
init_array2 = np.empty((samples, ts.numchannels),dtype=np.float32)
init_array3 = np.empty((samples, ts.numchannels),dtype=np.float32)

def print_number_of_blocks_in_block_buffers():
    buffers = list(ss.block_buffer.values())
    elements = [len(buf) for buf in buffers]
    print(dict(zip(['tp1','tp2','tp3'], elements)))

def do_stuff1(obj): # not time consuming function
    for _ in obj.result(2048): # 
        print("tp1 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.3) 

def do_stuff2(obj): # not time consuming function
    for i in obj.result(2048): # 
        print("tp2 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.5) 

def do_stuff3(obj): # more time consuming function
    for i in obj.result(2048): # 
        print("tp3 calls sample splitter")
        print_number_of_blocks_in_block_buffers()
        sleep(0.7) 

worker1 = threading.Thread(target=do_stuff1, args=(tp1,))
worker2 = threading.Thread(target=do_stuff2, args=(tp2,))
worker3 = threading.Thread(target=do_stuff3, args=(tp3,))

print("start threads")

worker1.start()
worker2.start()
worker3.start()

[thr.join() for thr in [worker1,worker2,worker3]]

print("threads finished")
