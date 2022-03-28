# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
""" Input from soundcard hardware using the SoundDevice library

.. autosummary::
    :toctree: generated/

    SoundDeviceSamplesGenerator
"""
import sounddevice as sd
from traits.api import Int, Long, Bool, Property, Any, observe, cached_property
from .tprocess import SamplesGenerator
from .internal import digest

class SoundDeviceSamplesGenerator(SamplesGenerator):

    """
    Controller for sound card hardware using sounddevice library

    Uses the device with index :attr:`device` to read samples
    from input stream, generates output stream via the generator
    :meth:`result`.
    """

    #: input device index, refers to sounddevice list
    device = Int(0, desc="input device index")

    #: Number of input channels, maximum depends on device
    numchannels = Long(1,
       desc="number of analog input channels that collects data")

    #: Number of samples to collect; defaults to -1.
    # If is set to -1 device collects till user breaks streaming by setting Trait: collectsamples = False
    numsamples = Long(-1,
        desc="number of samples to collect")    

    #: Indicates if samples are collected, helper trait to break result loop
    collectsamples = Bool(True,
        desc="Indicates if samples are collected")

    #: Sampling frequency of the signal, changes with sinusdevices
    sample_freq = Property(
        desc="sampling frequency")

    #: Indicates that the sounddevice buffer has overflown
    overflow = Bool(False,
        desc="Indicates if sounddevice buffer overflow")

    #: Indicates that the stream is collecting samples
    running = Bool(False,
        desc="Indicates that the stream is collecting samples")

    #: The sounddevice InputStream object for inspection
    stream = Any

    # internal identifier
    digest = Property( depends_on=['device', 'numchannels', 'numsamples'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)

    # checks that numchannels are not more than device can provide
    @observe('device,numchannels')
    def _get_numchannels( self, change ):
        self.numchannels = min(self.numchannels,sd.query_devices(self.device)['max_input_channels'])

    def _get_sample_freq( self ):
        return sd.query_devices(self.device)['default_samplerate']

    def device_properties( self ):
        """
        Returns
        -------
        Dictionary of device properties according to sounddevice
        """
        return sd.query_devices(self.device)

    def result(self, num):
        """
        Python generator that yields the output block-wise. Use at least a 
        block-size of one ring cache block. 
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        print(self.device_properties(),self.sample_freq)
        self.stream = stream_obj = sd.InputStream(
                device=self.device, channels=self.numchannels, 
                clip_off=True, samplerate=self.sample_freq)

        with stream_obj as stream:
            self.running = True
            if self.numsamples == -1: 
                while self.collectsamples: # yield data as long as collectsamples is True
                    data,self.overflow = stream.read(num)
                    yield data[:num]   
                                           
            elif self.numsamples > 0: # amount of samples to collect is specified by user            
                samples_count = 0 # numsamples counter 
                while samples_count < self.numsamples: 
                    anz = min(num, self.numsamples-samples_count)
                    data, self.overflow = stream.read(num)
                    yield data[:anz]   
                    samples_count += anz
        self.running = False
        return  
