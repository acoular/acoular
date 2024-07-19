Documentation
=============

If you spot any errors or have suggestions for improvements, please feel free to submit a pull request.
The package documentation is provided under ``acoular/docs/source``. See :ref:`Compile Documentation` for instructions on how to build the documentation.

The full documentation is built from different sources, which are combined in the final documentation using the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ package.

1. `User Documentation`_: This is located in the ``source`` directory. This directory contains the ``index.rst`` file which serves as the root document (landing page) embedding several other subdocuments (sub-pages) written in  `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format. 
2. `API Documentation`_: This is generated from the source code documentation using the Sphinx `autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_ extension. The API documentation can be found in the ``source/api_ref`` subdirectory of the documentation.


.. _User Documentation:

User Documentation
------------------

This is the easiest way to contribute to the documentation. You can simply edit the corresponding ``.rst`` files in the ``acoular/docs/source`` directory. Make sure you are using the `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format. If you want to add a new document, make sure to include it in the ``index.rst`` or any related file.

.. _API Documentation:

Documenting the API
-------------------

If you have added a new class, method, or function, it is required to add a docstring explaining its purpose and functionality.

Style Guide
~~~~~~~~~~~

Acoular documentation relies on the `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

An example of a class documentation is given below by a shortened version of the `acoular.sources.TimeSamples` class for reading time-data from HDF5 files.

.. code-block:: python

    class TimeSamples(SamplesGenerator):
        """Container for processing time data in `*.h5` or NumPy array format.

        This class loads measured data from HDF5 files and provides information about this data.
        It also serves as an interface where the data can be accessed (e.g. for use in a block chain) via the
        :meth:`result` generator.

        Examples
        --------
        Data can be loaded from a HDF5 file as follows:

        >>> from acoular import TimeSamples
        >>> ts = TimeSamples(name='path/filename.h5')

        Alternatively, the time data can be specified directly as a numpy array.
        In this case, the :attr:`data` and :attr:`sample_freq` attributes must be set manually.

        >>> from acoular import TimeSamples
        >>> import numpy as np
        >>> data = np.random.rand(1000, 4)
        >>> ts = TimeSamples(data=data, sample_freq=51200)

        Chunks of the time data can be accessed iteratively via the :meth:`result` generator:

        >>> blocksize = 256
        >>> generator = ts.result(num=blocksize)
        >>> for block in generator:
        ...     print(block.shape)

        See Also
        --------
        acoular.sources.MaskedTimeSamples :
            Extends the functionality of class :class:`TimeSamples` by enabling the definition of start and stop samples
            as well as the specification of invalid channels.
        """

        #: Full name of the .h5 file with data.
        name = File(filter=['*.h5'], desc='name of data file')

        def result(self, num=128):
            """Python generator that yields the output block-wise.

            Reads the time data either from a HDF5 file or from a numpy array given
            by :attr:`data` and iteratively returns a block of size `num` samples.
            Calibrated data is returned if a calibration object is given by :attr:`calib`.

            Parameters
            ----------
            num : integer, defaults to 128
                This parameter defines the size of the blocks to be yielded
                (i.e. the number of samples per block).

            Yields
            ------
            numpy.ndarray
                Samples in blocks of shape (num, numchannels).
                The last block may be shorter than num.

            """
            if self.numsamples == 0:
                msg = 'no samples available'
                raise OSError(msg)
            self._datachecksum  # trigger checksum calculation # noqa: B018
            i = 0
            if self.calib:
                if self.calib.num_mics == self.numchannels:
                    cal_factor = self.calib.data[newaxis]
                else:
                    raise ValueError('calibration data not compatible: %i, %i' % (self.calib.num_mics, self.numchannels))
                while i < self.numsamples:
                    yield self.data[i : i + num] * cal_factor
                    i += num
            else:
                while i < self.numsamples:
                    yield self.data[i : i + num]
                    i += num


* The class docstring contains a short summary in the first line, followed by an extended summary. An extended summary is not always necessary, but it is recommended for complex classes. Ideally, one includes a short code snippet in the extended summary
* The `name` attribute is documented using a comment line above the attribute definition starting with `#:`.
* The public `result` method includes a short summary line and the parameters and the return values are documented along with their types and shapes.
* Additional information can be included in the `See Also` section, which lists related classes or functions.

More information on the NumPy docstring format can be found in the `NumPy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.



Autosummary 
~~~~~~~~~~~

To ensure that a new module, class, or function is included in the API documentation, it needs to be added to the `autosummary` section at the top of the respective Python module file so that it can be recognized by Sphinx. The `autosummary` section is a comment block that is followed by the names of the classes and functions to be included.

