#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows the different behaviour of SampleSplitter class  
when the maximum size of a block buffer is reached
"""

from acoular import TimePower, MaskedTimeSamples, SampleSplitter
import threading
from time import sleep

samples = 25000
h5savefile = 'example_data.h5'
ts = MaskedTimeSamples( name=h5savefile,
                       start = 0,
                       stop = samples)  

# set up Sample Splitter
ss = SampleSplitter(source = ts)

tp1 = TimePower(source=ss)
tp2 = TimePower(source=ss)
ss.register_object(tp1,tp2) # register objects

def print_number_of_blocks_in_block_buffers():
    buffers = list(ss.block_buffer.values())
    elements = [len(buf) for buf in buffers]
    print(dict(zip(['tp1','tp2'], elements)))

def do_stuff1(obj): #
    for i in obj.result(2048): 
        print_number_of_blocks_in_block_buffers()
        sleep(0.1) 

def do_stuff2(obj): #
    for i in obj.result(2048): 
        print_number_of_blocks_in_block_buffers()
        sleep(0.8) 

#%%

print("buffer overflow behaviour == 'none'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'none'
ss.buffer_overflow_treatment[tp2] = 'none'

worker1 = threading.Thread(target=do_stuff1, args=(tp1,))
worker2 = threading.Thread(target=do_stuff2, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")

#%%

print("buffer overflow behaviour == 'warning'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'warning'
ss.buffer_overflow_treatment[tp2] = 'warning'

worker1 = threading.Thread(target=do_stuff1, args=(tp1,))
worker2 = threading.Thread(target=do_stuff2, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")

#%%
print("buffer overflow behaviour == 'error'")
print("buffer size is set to a maximum of 5 elements")
ss.buffer_size=5

ss.buffer_overflow_treatment[tp1] = 'error'
ss.buffer_overflow_treatment[tp2] = 'error'

worker1 = threading.Thread(target=do_stuff1, args=(tp1,))
worker2 = threading.Thread(target=do_stuff2, args=(tp2,))

print("start threads")

worker1.start()
worker2.start()

worker1.join()
worker2.join()

print("threads finished")