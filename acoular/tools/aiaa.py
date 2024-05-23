# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Classes for importing AIAA Array Benchmarks
from . import aiaa
These classes allow importing data from HDF5 files following the specifications of
the AIAA microphone array methods benchmarking effort:
https://www-docs.b-tu.de/fg-akustik/public/veroeffentlichungen/ArrayMethodsFileFormatsR2P4Release.pdf .

The classes are derived from according Acoular classes so that they can be used directly within
the framework.

Examples
--------
>>> micgeom = MicAIAABenchmark(name='some_benchmarkdata.h5')
>>> timedata = TimeSamplesAIAABenchmark(name='some_benchmarkdata.h5')


.. autosummary::
    :toctree: generated/

    TimeSamplesAIAABenchmark
    TriggerAIAABenchmark
    CsmAIAABenchmark
    MicAIAABenchmark
"""

from os import path

from numpy import array
from traits.api import (
    File,
    Instance,
    Property,
    cached_property,
    on_trait_change,
    property_depends_on,
)

from acoular.h5files import H5FileBase, _get_h5file_class
from acoular.internal import digest
from acoular.microphones import MicGeom
from acoular.sources import TimeSamples
from acoular.spectra import PowerSpectraImport


class TimeSamplesAIAABenchmark(TimeSamples):
    """Container for AIAA benchmark data in `*.h5` format.

    This class loads measured data from h5 files in AIAA benchmark format
    and and provides information about this data.
    Objects of this class behave similar to :class:`~acoular.sources.TimeSamples`
    objects.
    """

    def load_timedata(self):
        """Loads timedata from .h5 file. Only for internal use."""
        self.data = self.h5f.get_data_by_reference('MicrophoneData/microphoneDataPa')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sampleRateHz')
        (self.numsamples, self.numchannels) = self.data.shape

    def load_metadata(self):
        """Loads metadata from .h5 file. Only for internal use."""
        self.metadata = {}
        if '/MetaData' in self.h5f:
            self.metadata = self.h5f.node_to_dict('/MetaData')


class TriggerAIAABenchmark(TimeSamplesAIAABenchmark):
    """Container for tacho data in  `*.h5` format.

    This class loads tacho data from h5 files as specified in
    "Microphone Array Benchmark b11: Rotating Point Sources"
    (https://doi.org/10.14279/depositonce-8460)
    and and provides information about this data.
    """

    def load_timedata(self):
        """Loads timedata from .h5 file. Only for internal use."""
        self.data = self.h5f.get_data_by_reference('TachoData/tachoDataV')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sampleRateHz')
        (self.numsamples, self.numchannels) = self.data.shape


class CsmAIAABenchmark(PowerSpectraImport):
    """Class to load the CSM that is stored in AIAA Benchmark HDF5 file."""

    #: Full name of the .h5 file with data
    name = File(filter=['*.h5'], desc='name of data file')

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property(
        depends_on='name',
        desc='basename of data file',
    )

    #: number of channels
    numchannels = Property()

    #: HDF5 file object
    h5f = Instance(H5FileBase, transient=True)

    # internal identifier
    digest = Property(depends_on=['basename', '_csmsum'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return path.splitext(path.basename(self.name))[0]

    @on_trait_change('basename')
    def load_data(self):
        """Open the .h5 file and set attributes."""
        if not path.isfile(self.name):
            # no file there
            raise OSError('No such file: %s' % self.name)
        if self.h5f is not None:
            try:
                self.h5f.close()
            except OSError:
                pass
        file = _get_h5file_class()
        self.h5f = file(self.name)

    # @property_depends_on( 'block_size, ind_low, ind_high' )
    def _get_indices(self):
        try:
            return range(self.fftfreq().shape[0])  # [ self.ind_low: self.ind_high ]
        except IndexError:
            return range(0)

    @property_depends_on('digest')
    def _get_numchannels(self):
        try:
            attrs = self.h5f.get_data_by_reference('MetaData/ArrayAttributes')
            return self.h5f.get_node_attribute(attrs, 'microphoneCount')
        except IndexError:
            return 0

    @property_depends_on('digest')
    def _get_csm(self):
        """Loads cross spectral matrix from file."""
        csmre = self.h5f.get_data_by_reference('/CsmData/csmReal')[:].transpose((2, 0, 1))
        csmim = self.h5f.get_data_by_reference('/CsmData/csmImaginary')[:].transpose((2, 0, 1))
        csmdatagroup = self.h5f.get_data_by_reference('/CsmData')
        sign = self.h5f.get_node_attribute(csmdatagroup, 'fftSign')
        return csmre + sign * 1j * csmim

    def fftfreq(self):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return array(self.h5f.get_data_by_reference('/CsmData/binCenterFrequenciesHz')[:].flatten(), dtype=float)


class MicAIAABenchmark(MicGeom):
    """Provides the geometric arrangement of microphones in the array.

    In contrast to standard Acoular microphone geometries, the AIAA
    benchmark format includes the array geometry as metadata in the
    file containing the measurement data.
    """

    #: Name of the .h5-file from wich to read the data.
    from_file = File(filter=['*.h5'], desc='name of the h5 file containing the microphone geometry')

    @on_trait_change('basename')
    def import_mpos(self):
        """Import the microphone positions from .h5 file.
        Called when :attr:`basename` changes.
        """
        if not path.isfile(self.from_file):
            # no file there
            raise OSError('No such file: %s' % self.from_file)

        file = _get_h5file_class()
        h5f = file(self.from_file, mode='r')
        self.mpos_tot = h5f.get_data_by_reference('MetaData/ArrayAttributes/microphonePositionsM')[:].swapaxes(0, 1)
        h5f.close()
