.. _caching:

Caching
=======

Many Acoular workflows contain expensive intermediate results. For example, a
frequency-domain beamforming workflow may first read time data in blocks,
calculate a cross-spectral matrix, and then calculate beamforming maps for one
or more frequencies. Acoular avoids repeating such work by combining
:ref:`lazy_evaluation` with file-based caching.

Lazy evaluation decides when a calculation is needed. Caching decides whether
Acoular can reuse data from an earlier calculation. This chapter explains the
file-based caching controls.

The code snippets below are taken from
:doc:`../auto_examples/introductory_examples/example_caching`.

What Is Cached
--------------

The most important user-visible caches are HDF5 file caches. They are used by
classes that produce results that are expensive to calculate and useful to reuse:

* :class:`~acoular.spectra.PowerSpectra` can cache the cross-spectral matrix,
  eigenvalues, and eigenvectors.
* :class:`~acoular.fbeamform.BeamformerBase` and derived frequency-domain
  beamformers can cache beamforming results.
* :class:`~acoular.fbeamform.PointSpreadFunction` can cache point spread
  functions.
* :class:`~acoular.process.Cache` explicitly caches the block-wise output of any
  source object derived from :class:`~acoular.base.Generator`.

These caches are different from the original measurement files, such as HDF5
files read by :class:`~acoular.sources.TimeSamples`. Measurement data remains in
the original file and is read in blocks. Cache files contain calculated results
that can be regenerated from the source data and object settings.

Cache Files And Cache Keys
--------------------------

Cache files are written to the :attr:`~acoular.configuration.Config.cache_dir`
setting, which is available as ``ac.config.cache_dir``. If no cache directory is
set explicitly, Acoular creates and uses a directory named ``cache`` below the
current working directory when the cache is first needed.

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Configure cache directory, backend, and global mode.
   :end-before: # %%

The remaining examples on this page use the standard introductory setup:

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Define shared geometry and input data.
   :end-before: # %%

Cache files use the suffix ``_cache.h5``. Where possible, the file name is based
on the original source data. For example, processing data from
``three_sources.h5`` typically creates a cache file named
``three_sources_cache.h5``.

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Check the generated cache file.
   :end-before: # %%

Inside a cache file, Acoular stores results under names that include a digest.
The digest is calculated from the class and the traits that affect the result,
for example the block size of :class:`~acoular.spectra.PowerSpectra`, the source
digest, the steering vector, or beamformer settings. If such a trait changes,
the digest changes and Acoular calculates a new result instead of reusing an
outdated one.

Global Caching Modes
--------------------

The global cache behavior is controlled by the
:attr:`~acoular.configuration.Config.global_caching` setting, which is available
as ``ac.config.global_caching``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Meaning
   * - ``"individual"``
     - Default. Classes with a ``cached`` trait decide individually whether to
       use the file cache.
   * - ``"all"``
     - Use file caching wherever supported, even if an individual object has
       ``cached=False``.
   * - ``"none"``
     - Do not use file caching and do not create cache files.
   * - ``"readonly"``
     - Read existing cache data if it is available, but do not create or update
       cache files.
   * - ``"overwrite"``
     - Recalculate supported cached results and replace existing cache entries.

For day-to-day work, the default ``"individual"`` mode is usually the most
convenient. Use ``"none"`` when benchmarking or debugging and you want every
result to be recalculated. Use ``"readonly"`` when working with a shared or
archived cache directory that should not be modified. Use ``"overwrite"`` when
you deliberately want to refresh existing cache entries.

The following example shows ``"overwrite"`` in use:

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Force recalculation in overwrite mode.
   :end-before: # %%

Object-Level Caching
--------------------

Some classes also provide a ``cached`` trait. In the default
``global_caching="individual"`` mode, this trait controls whether that object
uses the file cache.

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Use object-level caching controls.
   :end-before: # %%
   
The object-level ``cached`` trait is ignored by the global ``"all"``,
``"none"``, ``"readonly"``, and ``"overwrite"`` modes. Those modes are intended
for session-wide control.

Explicitly Caching A Processing Chain
-------------------------------------

The :class:`~acoular.process.Cache` class is useful when a result is produced by
a block-wise generator and will be consumed more than once. It is often used in
time-domain processing chains, where repeating all upstream calculations would
be expensive.

.. literalinclude:: ../../examples/introductory_examples/example_caching.py
   :language: python
   :start-after: # Explicitly cache a time-domain processing chain.

If a :class:`~acoular.process.Cache` object finds an incomplete cache entry, it
continues safely according to the global cache mode. In normal writable modes,
it can rebuild or complete the cache. In ``"readonly"`` mode, incomplete cache
data is not modified.

Choosing The HDF5 Backend
-------------------------

Acoular can use either PyTables or h5py for HDF5 cache files. The backend is
selected with the :attr:`~acoular.configuration.Config.h5library` setting, which
is available as ``ac.config.h5library``.

Both backends write HDF5 files. PyTables is preferred when available; if it is
not installed, Acoular uses h5py.

Practical Tips
--------------

Keep cache files when you expect to revisit the same analysis with the same
settings. Remove them when disk space is more important than recalculation time.
Because cache files contain derived data, deleting them does not delete the
original measurement data.

Close the Python session, or make sure no Acoular objects are still using a cache
file, before deleting cache files manually. During a session, Acoular keeps open
HDF5 file handles for objects that reference cached data.

If a result does not seem to update after changing an input, first check whether
the input is part of the object's digest dependencies. Most user-facing traits
that affect a calculation are included. For a forced refresh, set
``ac.config.global_caching = "overwrite"`` for the calculation or delete the
corresponding cache file after the session has released it.
