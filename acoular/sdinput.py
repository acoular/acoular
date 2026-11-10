# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Input from soundcard hardware using the SoundDevice library.

.. autosummary::
    :toctree: generated/

    SoundDeviceSamplesGenerator
"""

import logging
from warnings import warn as _warn

# acoular imports
from .base import AcoularBase, SamplesGenerator
from .configuration import config

from traits.api import Any, Bool, Enum, Float, Int, Property, observe


def _import_sounddevice():
    if config.have_sounddevice is False:
        msg = 'SoundDevice library not found but is required for using the SoundDeviceSamplesGenerator class.'
        raise ImportError(msg)
    import sounddevice as sd

    return sd


class SoundDeviceSamplesGenerator(SamplesGenerator):
    """Controller for sound card hardware using sounddevice library.

    Uses the device with index :attr:`device` to read samples
    from input stream, generates output stream via the generator
    :meth:`result`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _import_sounddevice()

    #: input device index, refers to sounddevice list
    device = Int(0)

    #: Number of input channels, maximum depends on device
    num_channels = Int(1)

    #: Number of samples to collect; defaults to -1. If is set to -1 device collects until user
    # breaks streaming by setting Trait: collect_samples = False.
    num_samples = Int(-1)

    #: Indicates if samples are collected, helper trait to break result loop
    collect_samples = Bool(True)

    #: Sampling frequency of the signal, changes with sinusdevices
    sample_freq = Property()

    _sample_freq = Float(default_value=None)

    #: Datatype (resolution) of the signal, used as `dtype` in a sd `Stream` object
    precision = Enum('float32', 'float16', 'int32', 'int16', 'int8', 'uint8')

    #: Indicates whether the last sounddevice read reported an input overflow.
    overflow = Bool(False)

    #: Indicates that the stream is collecting samples
    running = Bool(False)

    #: The sounddevice InputStream object for inspection
    stream = Any

    #: A unique identifier for the generator, based on its properties. (read-only)
    digest = Property(fget=AcoularBase._get_digest, depends_on=['device', 'num_channels', 'num_samples'])

    # checks that num_channels are not more than device can provide
    @observe('device, num_channels')
    def _get_num_channels(self, event):  # noqa: ARG002
        sd = _import_sounddevice()
        self.num_channels = min(self.num_channels, sd.query_devices(self.device)['max_input_channels'])

    def _get_sample_freq(self):
        if self._sample_freq is None:
            sd = _import_sounddevice()
            self._sample_freq = sd.query_devices(self.device)['default_samplerate']
        return self._sample_freq

    def _set_sample_freq(self, f):
        self._sample_freq = f

    def device_properties(self):
        """
        Display the properties of the sounddevice input device.

        Returns
        -------
        Dictionary of device properties according to sounddevice
        """
        sd = _import_sounddevice()
        return sd.query_devices(self.device)

    def result(self, num):
        """
        Python generator that yields the output block-wise.

        Use at least a block-size of one ring cache block.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`).
            The last block may be shorter than num.

        Warns
        -----
        UserWarning
            On the first input overflow per recording. The final INFO summary counts
            flagged reads, not lost samples. Normal Python warning filters apply.

        """
        sd = _import_sounddevice()
        logger = self.logger
        if logger.isEnabledFor(logging.DEBUG):
            properties = self.device_properties()
            logger.debug(
                'Selected input device',
                device=self.device,
                name=properties.get('name'),
                max_channels=properties.get('max_input_channels'),
                default_sample_freq=properties.get('default_samplerate'),
            )
        self.stream = stream_obj = sd.InputStream(
            device=self.device,
            channels=self.num_channels,
            clip_off=True,
            samplerate=self.sample_freq,
            dtype=self.precision,
        )
        total = self.num_samples if self.num_samples >= 0 else None
        with logger.operation(
            'Recording samples',
            'Stopped recording',
            total=total,
            count='samples',
            stopped='Stopped recording',
            device=self.device,
            channels=self.num_channels,
            sample_freq=self.sample_freq,
            overflow_reads=0,
        ) as task:
            fields = task.fields
            try:
                with stream_obj as stream:
                    self.running = True
                    logger.info('Started recording', **fields)
                    samples_count = 0
                    while (self.num_samples == -1 and self.collect_samples) or samples_count < self.num_samples:
                        anz = num if self.num_samples == -1 else min(num, self.num_samples - samples_count)
                        data, self.overflow = stream.read(num)
                        if self.overflow:
                            fields['overflow_reads'] += 1
                            if fields['overflow_reads'] == 1:
                                _warn(
                                    f'Input overflow detected; recorded samples may be missing (device={self.device})',
                                    UserWarning,
                                    stacklevel=2,
                                )
                        data = data[:anz]
                        task.advance(data.shape[0])
                        samples_count += anz
                        yield data
            finally:
                self.running = False
            fields['reason'] = 'requested_stop' if self.num_samples == -1 else 'sample_limit'
