# coding=UTF-8
"""
import filter for nidaq-mx

nidaqimport.py (c) Ennes Sarradj 2008, all rights reserved
"""

__author__ = "Ennes Sarradj, ennes.sarradj@gmx.de"
__date__ = "17 April 2008"
__version__ = "1.0beta"


from beamfpy import TimeSamples, td_dir
from fileimport import time_data_import
import ctypes
import numpy
import time
import tables
from enthought.traits.api import HasTraits, HasPrivateTraits, Float, Int, File, CArray, Property, Instance, Trait, Bool, Any, List, Str, Long
from enthought.traits.ui.api import EnumEditor
from enthought.traits.ui.api import View, Item, Group
from enthought.traits.ui.menu import OKCancelButtons
from datetime import datetime
from os import path

nidaq = ctypes.windll.nicaiu # load the DLL
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
            buf_size = 128
            buf = ctypes.create_string_buffer('\000' * buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            buf1 = ctypes.create_string_buffer('\000' * buf_size)
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
DAQmxGetSampClkRate = ECFactory(nidaq.DAQmxGetSampClkRate)
DAQmxSetSampClkRate = ECFactory(nidaq.DAQmxSetSampClkRate)

class nidaq_import( time_data_import ):
    """
    import of data from measurement using NI-DAQmx
    """

    # name of the NI task to use
    taskname = Str(
        desc="name of the NI task to use for the measurement")

    # sample rate
    sample_freq = Float(48000.0,
        desc="sampling frequency")

    # number of time data samples
    numsamples = Long(48000,
        desc="number of samples")

    # number of channels, is set automatically
    numchannels =  Long(0,
        desc="number of channels in the task")

    # number of devices, is set automatically
    numdevices = Long(0,
        desc="number of devices in the task")

    # name of channels, is set automatically
    namechannels =  List(
        desc="names of channels in the task")

    # number of devices, is set automatically
    namedevices = List(
        desc="names of devices in the task")

    # name of available and valid tasks
    tasknames = List

    traits_view = View(
        [   Item('taskname{Task name}', editor = EnumEditor(name = 'tasknames')),
            ['sample_freq','numsamples','-'],
            [
                ['numdevices~{count}',Item('namedevices~{names}',height = 3),'-[Devices]'],
                ['numchannels~{count}',Item('namechannels~{names}',height = 3),'-[Channels]'],
            ],
            '|[Task]'
        ],
        title='NI-DAQmx data aquisition',
        buttons = OKCancelButtons
                    )

    def __init__ ( self, **traits ):
        time_data_import.__init__(self, **traits )
        taskHandle = TaskHandle(0)
        buf_size = 1024
        buf = ctypes.create_string_buffer('\000' * buf_size)
        DAQmxGetSysTasks(ctypes.byref(buf),buf_size)
        tasknamelist = buf.value.split(', ')
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
        buf = ctypes.create_string_buffer('\000' * buf_size)
        num = uInt32()
        fnum = float64()
        lnum = uInt64()
        try:
            DAQmxLoadTask(self.taskname,ctypes.byref(taskHandle))
        except RuntimeError:
            return
        DAQmxGetTaskNumChans(taskHandle,ctypes.byref(num))
        self.numchannels = num.value
        # commented for compatibility with older NIDAQmx
        #~ DAQmxGetTaskNumDevices(taskHandle,ctypes.byref(num))
        #~ self.numdevices = num.value
        DAQmxGetTaskChannels(taskHandle,ctypes.byref(buf),buf_size)
        self.namechannels = buf.value.split(', ')
        DAQmxGetTaskDevices(taskHandle,ctypes.byref(buf),buf_size)
        self.namedevices = buf.value.split(', ')
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
            DAQmxLoadTask(self.taskname,ctypes.byref(taskHandle))
        except RuntimeError:
            return
        try:
            DAQmxSetSampClkRate(taskHandle,float64(self.sample_freq))
        except RuntimeError:
            pass
        DAQmxGetSampClkRate(taskHandle,ctypes.byref(fnum))
        self.sample_freq = fnum.value
        DAQmxClearTask(taskHandle)
        print self.sample_freq


    def get_data (self,td):
        """
        main work is done here: imports the data from CSV file into
        TimeSamples object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        taskHandle = TaskHandle(0)
        read = uInt32()
        fnum = float64()
        lnum = uInt64()
        try:
            DAQmxLoadTask(self.taskname,ctypes.byref(taskHandle))
            if self.numchannels<1:
                raise RuntimeError
        except RuntimeError:
            # no valid task
            time_data_import.getdata(self,td)
            return
        #import data
        name = td.name
        if name=='':
            name = datetime.now().isoformat('_').replace(':','-').replace('.','_')
        name = path.join(td_dir,name+'.h5')
        f5h = tables.openFile(name,mode='w')
        ac = f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,self.numchannels))
        ac.setAttr('sample_freq',self.sample_freq)
        DAQmxGetSampQuantSampPerChan(taskHandle,ctypes.byref(lnum))
        max_num_samples = lnum.value
        data = numpy.empty((max_num_samples,self.numchannels),dtype=numpy.float64)
        DAQmxStartTask(taskHandle)
        count = 0L
        numsamples = self.numsamples
        while count<numsamples:
            DAQmxReadAnalogF64(taskHandle,-1,float64(10.0),
                                     DAQmx_Val_GroupByScanNumber,data.ctypes.data,
                                     data.size,ctypes.byref(read),None)
            ac.append(numpy.array(data[:min(read.value,numsamples-count)],dtype=numpy.float32))
            count+=read.value
        DAQmxStopTask(taskHandle)
        DAQmxClearTask(taskHandle)
        f5h.close()
        td.name = name
        td.load_data()

if __name__=='__main__':
    x=nidaq_import()
    x.configure_traits()
    td=TimeSamples()
    x.get_data(td)