# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Input from soundcard hardware using the SoundDevice library.

.. autosummary::
    :toctree: generated/

    SoundDeviceSamplesGenerator
"""

from traits.api import Any, Bool, Enum, Float, Int, Property, cached_property, observe

# acoular imports
from .base import SamplesGenerator
from .configuration import config
from .deprecation import deprecated_alias
from .internal import digest

if config.have_sounddevice:
    import sounddevice as sd


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples', 'collectsamples': 'collect_samples'})
class SoundDeviceSamplesGenerator(SamplesGenerator):
    """Controller for sound card hardware using sounddevice library.

    Uses the device with index :attr:`device` to read samples
    from input stream, generates output stream via the generator
    :meth:`result`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.have_sounddevice is False:
            msg = 'SoundDevice library not found but is required for using the SoundDeviceSamplesGenerator class.'
            raise ImportError(msg)

    #: input device index, refers to sounddevice list
    device = Int(0, desc='input device index')

    #: Number of input channels, maximum depends on device
    num_channels = Int(1, desc='number of analog input channels that collects data')

    #: Number of samples to collect; defaults to -1. If is set to -1 device collects until user
    # breaks streaming by setting Trait: collect_samples = False.
    num_samples = Int(-1, desc='number of samples to collect')

    #: Indicates if samples are collected, helper trait to break result loop
    collect_samples = Bool(True, desc='Indicates if samples are collected')

    #: Sampling frequency of the signal, changes with sinusdevices
    sample_freq = Property(desc='sampling frequency')

    _sample_freq = Float(default_value=None)

    #: Datatype (resolution) of the signal, used as `dtype` in a sd `Stream` object
    precision = Enum('float32', 'float16', 'int32', 'int16', 'int8', 'uint8', desc='precision (resolution) of signal')

    #: Indicates that the sounddevice buffer has overflown
    overflow = Bool(False, desc='Indicates if sounddevice buffer overflow')

    #: Indicates that the stream is collecting samples
    running = Bool(False, desc='Indicates that the stream is collecting samples')

    #: The sounddevice InputStream object for inspection
    stream = Any

    # internal identifier
    digest = Property(depends_on=['device', 'num_channels', 'num_samples'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    # checks that num_channels are not more than device can provide
    @observe('device, num_channels')
    def _get_num_channels(self, event):  # noqa ARG002
        self.num_channels = min(self.num_channels, sd.query_devices(self.device)['max_input_channels'])

    def _get_sample_freq(self):
        if self._sample_freq is None:
            self._sample_freq = sd.query_devices(self.device)['default_samplerate']
        return self._sample_freq

    def _set_sample_freq(self, f):
        self._sample_freq = f

    def device_properties(self):
        """Returns
        -------
        Dictionary of device properties according to sounddevice
        """
        return sd.query_devices(self.device)

    def result(self, num):
        """Python generator that yields the output block-wise. Use at least a
        block-size of one ring cache block.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`).
            The last block may be shorter than num.

        """
        print(self.device_properties(), self.sample_freq)
        self.stream = stream_obj = sd.InputStream(
            device=self.device,
            channels=self.num_channels,
            clip_off=True,
            samplerate=self.sample_freq,
            dtype=self.precision,
        )

        with stream_obj as stream:
            self.running = True
            if self.num_samples == -1:
                while self.collect_samples:  # yield data as long as collect_samples is True
                    data, self.overflow = stream.read(num)
                    yield data[:num]

            elif self.num_samples > 0:  # amount of samples to collect is specified by user
                samples_count = 0  # num_samples counter
                while samples_count < self.num_samples:
                    anz = min(num, self.num_samples - samples_count)
                    data, self.overflow = stream.read(num)
                    yield data[:anz]
                    samples_count += anz
        self.running = False
        return
