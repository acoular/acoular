# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Classes for importing AIAA Array Benchmarks.

These classes allow importing data from HDF5 files following the specifications of
the AIAA microphone array methods benchmarking effort:
https://www-docs.b-tu.de/fg-akustik/public/veroeffentlichungen/ArrayMethodsFileFormatsR2P4Release.pdf .

The classes are derived from according Acoular classes so that they can be used directly within
the framework.

Examples
--------
>>> micgeom = MicAIAABenchmark(file='some_benchmarkdata.h5')  # doctest: +SKIP
>>> timedata = TimeSamplesAIAABenchmark(file='some_benchmarkdata.h5')  # doctest: +SKIP


.. autosummary::
    :toctree: generated/

    TimeSamplesAIAABenchmark
    TriggerAIAABenchmark
    CsmAIAABenchmark
    MicAIAABenchmark
"""  # noqa: W505

import contextlib
from warnings import warn as _warn

from acoular.h5files import H5FileBase, _get_h5file_class
from acoular.internal import digest
from acoular.microphones import MicGeom
from acoular.sources import TimeSamples
from acoular.spectra import PowerSpectraImport
from acoular.tools.utils import get_file_basename

import numpy as np
from traits.api import (
    Bool,
    File,
    Instance,
    Property,
    Union,
    cached_property,
    observe,
    property_depends_on,
)


class TimeSamplesAIAABenchmark(TimeSamples):
    """Container for AIAA benchmark data in `*.h5` format.

    This class loads measured data from h5 files in AIAA benchmark format
    and and provides information about this data.
    Objects of this class behave similar to :class:`~acoular.sources.TimeSamples`
    objects.
    """

    #: Boolean flag, if 'True', time data in the h5 file will be interpreted as being shaped
    #: (:attr:`num_channels`, :attr:`num_samples`), allowing older revisions of the
    #: AIAA benchmark file format. Default is 'False', i.e. interpretation of data as shaped
    #: (:attr:`num_samples`, :attr:`num_channels`). This is automatically set when the h5 file
    #: is loaded or data is set.
    _data_transposed = Bool(False)

    def _get__datachecksum(self):
        if self._data_transposed:
            return self.data[0, :].sum()
        return self.data[:, 0].sum()

    @observe('data')
    def _load_shapes(self, event):  # noqa: ARG002
        # Set :attr:`num_channels` and :attr:`num_samples` from data.
        if self.data is not None:
            data_shape = self.data.shape
            self._data_transposed = (data_shape[0] < data_shape[1])
            if self._data_transposed:
                _warn(
                    f'Data is of shape ({data_shape[0]}, {data_shape[1]}) and may be stored as '
                    '(num_channels, num_samples). It will be transposed for further processing.',
                    Warning,
                    stacklevel=2
                )
                self.num_channels, self.num_samples = self.data.shape
            else:
                self.num_samples, self.num_channels = self.data.shape

    def _load_timedata(self):
        """Loads timedata from :attr:`.h5 file<file>`. Only for internal use."""
        self.data = self._h5f.get_data_by_reference('MicrophoneData/microphoneDataPa')
        self.sample_freq = self._h5f.get_node_attribute(self.data, 'sampleRateHz')

    def _load_metadata(self):
        """Loads :attr:`metadata` from :attr:`.h5 file<file>`. Only for internal use."""
        self.metadata = {}
        if '/MetaData' in self._h5f:
            self.metadata = self._h5f.node_to_dict('/MetaData')

    def result(self, num=128):
        """
        Generate blocks of time-domain data iteratively.

        The :meth:`result` method is a Python generator that yields blocks of time-domain data
        of the specified size. Data is either read from an HDF5 file (if :attr:`file` is set)
        or from a NumPy array (if :attr:`data` is directly provided).

        Parameters
        ----------
        num : :class:`int`, optional
            The size of each block to be yielded, representing the number of time-domain
            samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) representing a block of
            time-domain data. The last block may have fewer than ``num`` samples if the total number
            of samples is not a multiple of ``num``.

        Raises
        ------
        :obj:`OSError`
            If no samples are available (i.e., :attr:`num_samples` is ``0``).

        Examples
        --------
        Create a generator and access blocks of data:

        >>> import numpy as np
        >>> from acoular.aiaa import TimeSamplesAIAABenchmark
        >>> ts = TimeSamplesAIAABenchmark(data=np.random.rand(1000, 4), sample_freq=51200)
        >>> generator = ts.result(num=256)
        >>> for block in generator:
        ...     print(block.shape)
        (256, 4)
        (256, 4)
        (256, 4)
        (232, 4)

        Note that the last block may have fewer that ``num`` samples.
        """
        if self.num_samples == 0:
            msg = 'no samples available'
            raise OSError(msg)
        self._datachecksum  # trigger checksum calculation # noqa: B018
        i = 0
        if self._data_transposed:
            while i < self.num_samples:
                yield self.data[:, i : num + i].transpose()
                i += num
        else:
            while i < self.num_samples:
                yield self.data[i : num + i]
                i += num


class TriggerAIAABenchmark(TimeSamplesAIAABenchmark):
    """Container for tacho data in  `*.h5` format.

    This class loads tacho data from h5 files as specified in
    "Microphone Array Benchmark b11: Rotating Point Sources"
    (https://doi.org/10.14279/depositonce-8460)
    and and provides information about this data.
    """

    def _load_timedata(self):
        """Loads timedata from .h5 file. Only for internal use."""
        self.data = self._h5f.get_data_by_reference('TachoData/tachoDataV')
        self.sample_freq = self._h5f.get_node_attribute(self.data, 'sampleRateHz')
        (self.num_samples, self.num_channels) = self.data.shape


class CsmAIAABenchmark(PowerSpectraImport):
    """Class to load the CSM that is stored in AIAA Benchmark HDF5 file."""

    #: Full name of the .h5 file with data
    file = Union(None, File(filter=['*.h5'], exists=True), desc='name of data file')

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property(
        depends_on=['file'],
        desc='basename of data file',
    )

    #: number of channels
    num_channels = Property()

    #: HDF5 file object
    _h5f = Instance(H5FileBase, transient=True)

    #: A unique identifier for the CSM importer, based on its properties. (read-only)
    digest = Property(depends_on=['basename', '_csmsum'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return get_file_basename(self.file)

    @observe('basename')
    def _load_data(self, event):  # noqa: ARG002
        """Open the .h5 file and set attributes."""
        if self._h5f is not None:
            with contextlib.suppress(OSError):
                self._h5f.close()
        file = _get_h5file_class()
        self._h5f = file(self.file)

    # @property_depends_on( 'block_size, ind_low, ind_high' )
    def _get_indices(self):
        try:
            return range(self.fftfreq().shape[0])  # [ self.ind_low: self.ind_high ]
        except IndexError:
            return range(0)

    @property_depends_on(['digest'])
    def _get_num_channels(self):
        try:
            attrs = self._h5f.get_data_by_reference('MetaData/ArrayAttributes')
            return self._h5f.get_node_attribute(attrs, 'microphoneCount')
        except IndexError:
            return 0

    @property_depends_on(['digest'])
    def _get_csm(self):
        """Loads cross spectral matrix from file."""
        csmre = self._h5f.get_data_by_reference('/CsmData/csmReal')[:].transpose((2, 0, 1))
        csmim = self._h5f.get_data_by_reference('/CsmData/csmImaginary')[:].transpose((2, 0, 1))
        csmdatagroup = self._h5f.get_data_by_reference('/CsmData')
        sign = self._h5f.get_node_attribute(csmdatagroup, 'fftSign')
        return csmre + sign * 1j * csmim

    def fftfreq(self):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return np.array(self._h5f.get_data_by_reference('/CsmData/binCenterFrequenciesHz')[:].flatten(), dtype=float)


class MicAIAABenchmark(MicGeom):
    """Provides the geometric arrangement of microphones in the array.

    In contrast to standard Acoular microphone geometries, the AIAA
    benchmark format includes the array geometry as metadata in the
    file containing the measurement data.
    """

    #: Name of the .h5-file from which to read the data.
    file = Union(
        None, File(filter=['*.h5'], exists=True), desc='name of the h5 file containing the microphone geometry'
    )

    @observe('file')
    def _import_mpos(self, event):  # noqa: ARG002
        """
        Import the microphone positions from .h5 file.

        Called when :attr:`basename` changes.
        """
        file = _get_h5file_class()
        h5f = file(self.file, mode='r')
        self.pos_total = h5f.get_data_by_reference('MetaData/ArrayAttributes/microphonePositionsM')[:].swapaxes(0, 1)
        h5f.close()
