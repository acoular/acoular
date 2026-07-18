.. _lazy_evaluation:

Lazy Evaluation
===============

Acoular uses lazy evaluation throughout its processing objects. Creating an
object usually stores configuration only. The expensive calculation starts later,
when a result is actually requested.

This is important for acoustic array processing because a processing setup often
contains large measurement files, cross-spectral matrices, steering vectors,
beamforming maps, and time-domain processing chains. Lazy evaluation lets you
assemble and adjust such a setup without repeatedly reading data or recalculating
intermediate results.

The code snippets below are taken from
:doc:`../auto_examples/introductory_examples/example_basic_beamforming` and
:doc:`../auto_examples/introductory_examples/example_lazy_evaluation`.

The Basic Idea
--------------

Most Acoular objects describe one processing step and hold references to their
input objects. The objects form a graph or chain, but the graph is idle until a
result is pulled from it.

.. literalinclude:: ../../examples/introductory_examples/example_basic_beamforming.py
   :language: python
   :start-after: # Import packages.
   :end-before: # Calculate beamforming result.

At this point, the time data is not loaded into memory and the cross-spectral
matrix has not been calculated. The :class:`~acoular.sources.TimeSamples` object
knows where the data is, and the :class:`~acoular.spectra.PowerSpectra` object
knows how the spectra should be calculated.

The calculation starts when a result property, method, or generator output is
requested:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Request the cross-spectral matrix.
   :end-before: # Request a single frequency line from the lazy result object.

This access asks :class:`~acoular.spectra.PowerSpectra` for its cross-spectral
matrix. Only then does Acoular read the source data block by block and calculate
the matrix.

Lazy Traits And Generators
--------------------------

Acoular uses two common lazy patterns:

* Result properties and methods, for example
  :attr:`~acoular.spectra.PowerSpectra.csm` or
  :meth:`~acoular.fbeamform.BeamformerBase.synthetic`, calculate the requested
  result on demand.
* ``result(num)`` methods return Python generators. Calling ``result(num)``
  creates the generator; the processing starts when the generator is iterated,
  for example by calling :func:`next` or by using a ``for`` loop.

The second pattern is used heavily in time-domain processing. Each object pulls
blocks from its source only when its own output block is requested.

Example: Frequency-Domain Beamforming
-------------------------------------

The frequency-domain beamforming workflow in the getting-started tutorial is a
good example. First, define the input data and spectral processing, then define
the focus grid, microphone geometry, and steering vector. No beamforming map has
been calculated yet.

The beamformer result itself is also lazy per frequency. Accessing
``bb.result`` prepares a lazy result object. Individual frequency lines are
calculated only when they are indexed:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Request a single frequency line from the lazy result object.
   :end-before: # Trigger the beamforming map calculation.

In the usual workflow, the first full calculation is triggered by asking for a
map at a frequency or frequency band:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Trigger the beamforming map calculation.
   :end-before: # Change a parameter and request the result again.

This single request pulls the required upstream results:

* :class:`~acoular.spectra.PowerSpectra` calculates the cross-spectral matrix if
  it is not already available.
* :class:`~acoular.fbeamform.BeamformerBase` calculates the frequency lines
  needed for the requested third-octave band.
* The return value is reshaped to the grid shape and contains the beamforming
  map for that band.

If file caching is active, calculated frequency-domain results can also be
stored and reused. Lazy evaluation decides *when* a calculation is needed;
caching decides whether a previously calculated result can be loaded instead of
being recalculated. See :ref:`caching` for the cache controls.

Example: Time-Domain Beamforming Chain
--------------------------------------

Time-domain processing chains are lazy in a slightly different way. The output
is normally a stream of blocks, and each block is produced only when the final
consumer asks for it.

The following chain performs time-domain delay-and-sum beamforming with
:class:`~acoular.tbeamform.BeamformerTime`. The beamformer produces one time
signal per grid point. The chain then filters the beamformer output, squares it,
and averages it block-wise to create source maps.

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Build a lazy time-domain processing chain.
   :end-before: # Create the generator.

The chain is now configured, but no samples have been processed. Even the next
line only creates a generator object:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Create the generator.
   :end-before: # Pull the first block from the generator.

The first processing work starts when output is pulled from the generator:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Pull the first block from the generator.
   :end-before: # Add explicit caching to the lazy chain.

To create ``first_map``, Acoular pulls one averaged block from
:class:`~acoular.process.Average`. That object pulls the required number of
blocks from :class:`~acoular.tprocess.TimePower`, which pulls from
:class:`~acoular.tprocess.FiltOctave`, which pulls from
:class:`~acoular.tbeamform.BeamformerTime`, which finally pulls time samples
from ``ts``.

This pull-based model keeps memory use bounded for long recordings. The full
time history does not need to be present in memory at once.

Adding A Cache To A Lazy Chain
------------------------------

Lazy evaluation and caching are often used together in time-domain workflows.
Adding :class:`~acoular.process.Cache` at the end of the chain stores blocks
while they are pulled for the first time:

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Add explicit caching to the lazy chain.

The chain is still lazy. The cache does not force calculation when it is created;
it only changes what happens when blocks are requested. On the first pass,
missing blocks are calculated and written to the cache. On later passes, cached
blocks can be read instead.

Changing Parameters
-------------------

Because Acoular objects are trait based, changing a relevant trait changes the
state of the processing object. The next request uses the new settings and
recalculates the affected result.

.. literalinclude:: ../../examples/introductory_examples/example_lazy_evaluation.py
   :language: python
   :start-after: # Change a parameter and request the result again.
   :end-before: # Build a lazy time-domain processing chain.

Here, changing ``ps.block_size`` changes the spectral processing. The next
request for ``bb.synthetic`` must use a cross-spectral matrix calculated with the
new block size, so the affected downstream result is recalculated or looked up
under a different cache key.

Practical Consequences
----------------------

Lazy evaluation has a few practical effects:

* Creating Acoular objects is usually cheap; requesting results is where the
  work happens.
* The first result request may take much longer than later requests if
  intermediate data has already been calculated or cached.
* In time-domain chains, processing follows the final consumer. If no one
  iterates over the last ``result(num)`` generator, the upstream chain remains
  idle.
* When debugging, it is useful to identify the line that first asks for data:
  often a property access, a call to ``synthetic`` or ``integrate``, or a loop
  over ``result(num)``.
