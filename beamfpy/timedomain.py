# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:25:10 2010

@author: sarradj
"""
# imports from other packages
from numpy import *
from enthought.traits.api import HasPrivateTraits, Float, Int, Long,\
File, CArray, Property, Instance, Trait, Bool, Range, Delegate, Any, Str,\
cached_property, on_trait_change, property_depends_on
from enthought.traits.ui.api import View, Item
from enthought.traits.ui.menu import OKCancelButtons
from os import path
import tables
# beamfpy imports
from internal import digest

class SamplesGenerator( HasPrivateTraits ):
    """
    Base class for any generating signal processing block,
    generates output via the generator 'result'
    """

    # sample_freq of signal
    sample_freq = Float(1.0,
        desc="sampling frequency")
    
    # number of channels 
    numchannels = Long
               
    # number of samples 
    numsamples = Long
    
    # internal identifier
    digest = ''
               
    # result generator: delivers output in blocks of num samples
    def result(self,num):
        pass

class TimeSamples( SamplesGenerator ):
    """
    Container for time data, loads time data
    and provides information about this data
    """

    # full name of the .h5 file with data
    name = File(filter=['*.h5'],
        desc="name of data file")

    # basename of the .h5 file with data
    basename = Property( depends_on = 'name',#filter=['*.h5'],
        desc="basename of data file")
    
    # sampling frequency of the data, is set automatically
    sample_freq = Float(1.0,
        desc="sampling frequency")

    # number of channels, is set automatically
    numchannels = Long(0L,
        desc="number of input channels")

    # number of time data samples, is set automatically
    numsamples = Long(0L,
        desc="number of samples")

    # the time data as (numsamples,numchannels) array of floats
    data = Any(
        desc="the actual time data array")

    # hdf5 file object
    h5f = Instance(tables.File)
    
    # internal identifier
    digest = Property( depends_on = ['basename',])

    traits_view = View(
        ['name{File name}',
            ['sample_freq~{Sampling frequency}',
            'numchannels~{Number of channels}',
            'numsamples~{Number of samples}',
            '|[Properties]'],
            '|'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]
    
    @on_trait_change('basename')
    def load_data( self ):
        """ open the .h5 file and setting attributes
        """
        if not path.isfile(self.name):
            # no file there
            self.numsamples = 0
            self.numchannels = 0
            self.sample_freq = 0
            return None
        if self.h5f!=None:
            try:
                self.h5f.close()
            except:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples,self.numchannels) = self.data.shape
#        self.basename = path.basename(self.name)

    # generator function
    def result(self,n=128):
        i = 0
#        cal_factor = 1.0
#        if self.calib:
#            if self.calib.num_mics==self.numchannels:
#                cal_factor = self.calib.data[newaxis,:]
#            else:
#                cal_factor = 1.0
        while i<self.numsamples:
            yield self.data[i:i+n]#[:,self.channels]*cal_factor
            i += n

