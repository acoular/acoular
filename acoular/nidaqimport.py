# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

"""nidaqimport.py: interface to nidaq mx."""
import ctypes
from datetime import datetime, timezone
from os import path

import numpy as np
import tables
from traits.api import Float, List, Long, Str

from .fileimport import time_data_import
from .h5cache import td_dir
from .sources import TimeSamples

try:
    nidaq = ctypes.windll.nicaiu # load the DLL
except:
    raise ImportError
# type definitions
int32 = ctypes.c_long
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
float64 = ctypes.c_double
TaskHandle = uInt32

# DAQmx constants
DAQmx_Val_Cfg_Default = int32(-1)
DAQmx_Val_GroupByChannel = 0
DAQmx_Val_GroupByScanNumber = 1
DAQmx_Val_FiniteSamps = 10178 # Acquire or generate a finite number of samples.
DAQmx_Val_ContSamps = 10123 # Acquire or generate samples until you stop the task.
DAQmx_Val_HWTimedSinglePoint = 12522 # Acquire or generate samples continuously using hardware timing without a buffer. Hardware timed single point sample mode is supported only for the sample clock and change detection timing types.

##############################

def ECFactory(func):
    def func_new(*args):
        err = func(*args)
        if err < 0:
            buf_size = 256
#            buf = ctypes.create_string_buffer('\000' * buf_size)
            buf = ctypes.create_string_buffer(b'\000' * buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
#            buf1 = ctypes.create_string_buffer('\000' * buf_size)
            buf1 = ctypes.create_string_buffer(b'\000' * buf_size)

##            nidaq.DAQmxGetExtendedErrorInfo(ctypes.byref(buf1),buf_size)
##            print buf1.value
            raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))
    return func_new

DAQmxGetSysTasks = ECFactory(nidaq.DAQmxGetSysTasks)
DAQmxLoadTask = ECFactory(nidaq.DAQmxLoadTask)
DAQmxClearTask = ECFactory(nidaq.DAQmxClearTask)
DAQmxStartTask = ECFactory(nidaq.DAQmxStartTask)
DAQmxStopTask = ECFactory(nidaq.DAQmxStopTask)
DAQmxGetTaskDevices = ECFactory(nidaq.DAQmxGetTaskDevices)
#DAQmxGetTaskNumDevices = ECFactory(nidaq.DAQmxGetTaskNumDevices)
DAQmxGetTaskNumChans = ECFactory(nidaq.DAQmxGetTaskNumChans)
DAQmxGetTaskChannels = ECFactory(nidaq.DAQmxGetTaskChannels)
DAQmxGetBufInputBufSize = ECFactory(nidaq.DAQmxGetBufInputBufSize)
DAQmxReadAnalogF64 = ECFactory(nidaq.DAQmxReadAnalogF64)
DAQmxIsTaskDone = ECFactory(nidaq.DAQmxIsTaskDone)
DAQmxGetSampQuantSampMode = ECFactory(nidaq.DAQmxGetSampQuantSampMode)
DAQmxGetSampQuantSampPerChan = ECFactory(nidaq.DAQmxGetSampQuantSampPerChan)
DAQmxSetSampQuantSampPerChan = ECFactory(nidaq.DAQmxSetSampQuantSampPerChan)
DAQmxGetSampClkRate = ECFactory(nidaq.DAQmxGetSampClkRate)
DAQmxSetSampClkRate = ECFactory(nidaq.DAQmxSetSampClkRate)

class nidaq_import( time_data_import ):
    """Provides an interface to import of measurement data
    using NI-DAQmx.
    """

    #: Name of the NI task to use
    taskname = Str(
        desc="name of the NI task to use for the measurement")

    #: Sampling frequency, defaults to 48000.
    sample_freq = Float(48000.0,
        desc="sampling frequency")

    #: Number of time data samples, defaults to 48000.
    numsamples = Long(48000,
        desc="number of samples")

    #: Number of channels; is set automatically.
    numchannels =  Long(0,
        desc="number of channels in the task")

    #: Number of devices; is set automatically.
    numdevices = Long(0,
        desc="number of devices in the task")

    #: Name of channels; is set automatically.
    namechannels =  List(
        desc="names of channels in the task")

    #: Name of devices; is set automatically.
    namedevices = List(
        desc="names of devices in the task")

    #: Name of available and valid tasks.
    tasknames = List

    def __init__ ( self, **traits ):
        time_data_import.__init__(self, **traits )
        taskHandle = TaskHandle(0)
        buf_size = 1024
#        buf = ctypes.create_string_buffer('\000' * buf_size)
        buf = ctypes.create_string_buffer(b'\000' * buf_size)
        DAQmxGetSysTasks(ctypes.byref(buf),buf_size)
#        tasknamelist = buf.value.split(', ')
        tasknamelist = buf.value.split(b', ')

        self.tasknames=[]
        for taskname in tasknamelist:
            # is task valid ?? try to load
            try:
                DAQmxLoadTask(taskname,ctypes.byref(taskHandle))
            except RuntimeError:
                continue
            self.tasknames.append(taskname)
            DAQmxClearTask(taskHandle)

    def _taskname_changed ( self ):
        taskHandle = TaskHandle(0)
        buf_size = 1024*4
#        buf = ctypes.create_string_buffer('\000' * buf_size)
        buf = ctypes.create_string_buffer(b'\000' * buf_size)

        num = uInt32()
        fnum = float64()
        lnum = uInt64()
        try:
            DAQmxLoadTask(str.encode(self.taskname),ctypes.byref(taskHandle))
        except RuntimeError:
            return
        DAQmxGetTaskNumChans(taskHandle,ctypes.byref(num))
        self.numchannels = num.value
        # commented for compatibility with older NIDAQmx
        #~ DAQmxGetTaskNumDevices(taskHandle,ctypes.byref(num))
        #~ self.numdevices = num.value
        DAQmxGetTaskChannels(taskHandle,ctypes.byref(buf),buf_size)
        self.namechannels = buf.value.decode().split(', ')
        DAQmxGetTaskDevices(taskHandle,ctypes.byref(buf),buf_size)
        self.namedevices = buf.value.decode().split(', ')
        self.numdevices = len(self.namedevices)
        DAQmxGetSampClkRate(taskHandle,ctypes.byref(fnum))
        self.sample_freq = fnum.value
        DAQmxGetSampQuantSampMode(taskHandle,ctypes.byref(num))
        if num.value==DAQmx_Val_FiniteSamps:
            DAQmxGetSampQuantSampPerChan(taskHandle,ctypes.byref(lnum))
            self.numsamples = lnum.value
        DAQmxClearTask(taskHandle)

    def _sample_freq_changed(self,dispatch='ui'):
        taskHandle = TaskHandle(0)
        fnum = float64()
        try:
            DAQmxLoadTask(str.encode(self.taskname),ctypes.byref(taskHandle))
        except RuntimeError:
            return
        try:
            DAQmxSetSampClkRate(taskHandle,float64(self.sample_freq))
        except RuntimeError:
            pass
        DAQmxGetSampClkRate(taskHandle,ctypes.byref(fnum))
        self.sample_freq = fnum.value
        DAQmxClearTask(taskHandle)
        print(self.sample_freq)


    def get_data (self, td):
        """Main work is done here: loads data from buffer into
        :class:`~acoular.sources.TimeSamples` object `td` and saves also a
        '*.h5' file.
        """
        taskHandle = TaskHandle(0)
        read = uInt32()
        fnum = float64()
        lnum = uInt64()
        try:
            DAQmxLoadTask(str.encode(self.taskname),ctypes.byref(taskHandle))
            if self.numchannels<1:
                raise RuntimeError
            DAQmxSetSampClkRate(taskHandle,float64(self.sample_freq))
        except RuntimeError:
            # no valid task
            time_data_import.get_data(self,td)
            return
        #import data
        name = td.name
        if name=='':
            name = datetime.now(tz=timezone.utc).isoformat('_').replace(':','-').replace('.','_')
            name = path.join(td_dir,name+'.h5')
        f5h = tables.open_file(name,mode='w')
        ac = f5h.create_earray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,self.numchannels))
        ac.set_attr('sample_freq',self.sample_freq)
        DAQmxSetSampQuantSampPerChan(taskHandle,uInt64(100000))
        DAQmxGetSampQuantSampPerChan(taskHandle,ctypes.byref(lnum))
        max_num_samples = lnum.value
        print("Puffergroesse: %i" % max_num_samples)
        data = np.empty((max_num_samples,self.numchannels),dtype=np.float64)
        DAQmxStartTask(taskHandle)
        count = 0
        numsamples = self.numsamples
        while count<numsamples:
            #~ DAQmxReadAnalogF64(taskHandle,-1,float64(10.0),
                                     #~ DAQmx_Val_GroupByScanNumber,data.ctypes.data,
                                     #~ data.size,ctypes.byref(read),None)
            DAQmxReadAnalogF64(taskHandle,1024,float64(10.0),
                                     DAQmx_Val_GroupByScanNumber,data.ctypes.data,
                                     data.size,ctypes.byref(read),None)
            ac.append(np.array(data[:min(read.value,numsamples-count)],dtype=np.float32))
            count+=read.value
            #~ if read.value>200:
                #~ print count, read.value
        DAQmxStopTask(taskHandle)
        DAQmxClearTask(taskHandle)
        f5h.close()
        td.name = name
        td.load_data()

    def get_single (self):
        """Gets one block of data."""
        taskHandle = TaskHandle(0)
        read = uInt32()
        fnum = float64()
        lnum = uInt64()
        try:
            DAQmxLoadTask(str.encode(self.taskname),ctypes.byref(taskHandle))
            if self.numchannels<1:
                raise RuntimeError
        except RuntimeError:
            # no valid task
            time_data_import.get_data(self,td)
            return None
        #import data
        ac = np.empty((self.numsamples,self.numchannels),np.float32)
        DAQmxGetSampQuantSampPerChan(taskHandle,ctypes.byref(lnum))
        max_num_samples = lnum.value
        data = np.empty((max_num_samples,self.numchannels),dtype=np.float64)
        DAQmxStartTask(taskHandle)
        count = 0
        numsamples = self.numsamples
        while count<numsamples:
            DAQmxReadAnalogF64(taskHandle,-1,float64(10.0),
                                     DAQmx_Val_GroupByScanNumber,data.ctypes.data,
                                     data.size,ctypes.byref(read),None)
            anz = min(read.value,numsamples-count)
            ac[count:count+anz]=np.array(data[:anz],dtype=np.float32)
            count+=read.value
        DAQmxStopTask(taskHandle)
        DAQmxClearTask(taskHandle)
        return ac


if __name__=='__main__':
    x=nidaq_import()
    x.taskname = 'test1'
    x.configure_traits()
    td=TimeSamples()
    x.get_data(td)
