.. _notch_filtering:

Notch filtering of tonal noise
==============================

Microphone array measurements are often contaminated by strong tonal
components. A typical case is a drone or a propeller-driven aircraft carrying
its own array: the blade-passing frequency (BPF) and its harmonics dominate the
spectrum and mask the sources one actually wants to map. Notch filters remove
these tones while leaving the rest of the spectrum -- and, crucially, the phase
relations between the channels -- intact.

Acoular provides four related classes in :mod:`acoular.tprocess`:

.. list-table::
    :widths: 30 70
    :header-rows: 1

    * - Class
      - Use it when
    * - :class:`~acoular.tprocess.NotchFilter`
      - a single tone at a known, fixed frequency has to go
    * - :class:`~acoular.tprocess.AdaptiveNotchFilter`
      - the tone moves, e.g. because the rotational speed changes
    * - :class:`~acoular.tprocess.ZeroPhaseNotchFilter`
      - phase must be preserved, e.g. ahead of beamforming
    * - :class:`~acoular.tprocess.CascadeNotchFilter`
      - a whole harmonic series, possibly from several sources, has to go

The classes build on each other: :class:`~acoular.tprocess.AdaptiveNotchFilter`
extends :class:`~acoular.tprocess.NotchFilter`,
:class:`~acoular.tprocess.ZeroPhaseNotchFilter` extends
:class:`~acoular.tprocess.AdaptiveNotchFilter`, and
:class:`~acoular.tprocess.CascadeNotchFilter` composes a bank of them. The
implementation follows :cite:`Harvey2019`.

A complete, runnable walk-through is available in the
:doc:`notch filter example <../auto_examples/io_and_signal_processing_examples/example_notch_filter>`.

The notch and its pole radius
-----------------------------

:class:`~acoular.tprocess.NotchFilter` is a second-order IIR section that places
a pair of zeros directly on the unit circle at
:attr:`~acoular.tprocess.NotchFilter.f_notch` and a pair of poles just inside
it, at :attr:`~acoular.tprocess.NotchFilter.pole_radius`:

.. math::

    H(z) = \frac{1 - 2\cos(\theta)z^{-1} + z^{-2}}
                {1 - 2r\cos(\theta)z^{-1} + r^2 z^{-2}},
    \qquad \theta = 2\pi f_\mathrm{notch} / f_s

Because the zeros sit exactly on the unit circle, attenuation at
:math:`f_\mathrm{notch}` is in principle complete. The pole radius :math:`r`
controls how narrow the notch is; its -3 dB bandwidth is approximately

.. math::

    B \approx (1 - r)\, f_s / \pi

Two consequences are worth keeping in mind:

* **The bandwidth scales with the sampling frequency.** At
  :math:`f_s = 25.6` kHz, a pole radius of ``0.99`` already gives a notch about
  80 Hz wide, which may well swallow neighbouring harmonics. A radius of
  ``0.999`` narrows it to some 8 Hz. A radius that works well at one sampling
  rate is not automatically a good choice at another.
* **A narrow notch settles slowly.** The filter state needs roughly
  :math:`7 / |\ln r|` samples to settle -- about 7000 samples (0.27 s at
  25.6 kHz) for ``0.999``. Until then, the residual at the notch frequency is
  dominated by the start-up transient rather than by the depth of the notch.
  Discard or ignore the settling region when evaluating suppression.

.. code-block:: python

    import acoular as ac

    ts = ac.TimeSamples(file='measurement.h5')
    notch = ac.NotchFilter(source=ts, f_notch=200.0, pole_radius=0.999)

Preserving phase
----------------

A causal IIR notch introduces a phase shift that swings through a full
transition across the notch frequency. For beamforming this is harmful, because
the phase relations between the channels carry the source location.

:class:`~acoular.tprocess.ZeroPhaseNotchFilter` filters forwards and then
backwards, so the two phase shifts cancel. The combined response is
:math:`|H|^2`: real, non-negative, and therefore exactly zero-phase at every
frequency. The boundary states are set following :cite:`Gustafsson1996`, the
same approach :func:`scipy.signal.filtfilt` uses -- and for a fixed notch
frequency in :attr:`~acoular.tprocess.ZeroPhaseNotchFilter.batch_mode` the
result is numerically identical to it.

The price is that the filter is non-causal: it needs future samples.

* :attr:`~acoular.tprocess.ZeroPhaseNotchFilter.batch_mode` = ``True`` collects
  the whole signal before processing. Use it for offline analysis and whenever
  exact :func:`~scipy.signal.filtfilt` equivalence matters.
* By default the filter streams, buffering
  :attr:`~acoular.tprocess.ZeroPhaseNotchFilter.num_lookahead_blocks` blocks of
  future context for the backward pass. This introduces a latency of that many
  blocks. If the buffer is shorter than the settling time, the backward pass
  cannot settle and the class emits a warning -- either increase
  ``num_lookahead_blocks``, enlarge the block size, or switch to batch mode.

.. code-block:: python

    zp = ac.ZeroPhaseNotchFilter(
        source=ts, f_notch=200.0, pole_radius=0.999, batch_mode=True,
    )

Tracking a moving tone
----------------------

Rotational speed rarely stays constant, so the BPF drifts.
:class:`~acoular.tprocess.AdaptiveNotchFilter` follows it in one of two modes,
selected via :attr:`~acoular.tprocess.AdaptiveNotchFilter.mode`.

**External mode** (``mode='external'``) takes the frequency from
:attr:`~acoular.tprocess.AdaptiveNotchFilter.freq_source`, typically derived
from measured RPM data. The frequency source is an ordinary
:class:`~acoular.base.SamplesGenerator` that yields one frequency value per
sample, in lockstep with the signal blocks -- so a
:class:`~acoular.sources.TimeSamples` object holding the trajectory is enough:

.. code-block:: python

    rpm_source = ac.TimeSamples(data=freq_trajectory[:, np.newaxis], sample_freq=fs)
    adaptive = ac.AdaptiveNotchFilter(
        source=ts, freq_source=rpm_source, mode='external', pole_radius=0.995,
    )

Both generators must yield blocks of the same size; a mismatch raises a
:obj:`ValueError`.

**Auto mode** (``mode='auto'``) needs no reference. It initialises from an FFT
peak near :attr:`~acoular.tprocess.NotchFilter.f_notch` and then tracks the tone
with a normalised LMS update of the recursive gradient after
:cite:`Nehorai1985`. Should the estimate drift further than a quarter of the
notch bandwidth away from its starting point, a global FFT search re-locks the
filter, following :cite:`TanJiang2009`.

.. code-block:: python

    auto = ac.AdaptiveNotchFilter(
        source=ts, f_notch=195.0, mode='auto',
        mu=0.02, gradient_leak=0.95, pole_radius=0.99,
    )

Two knobs matter most: :attr:`~acoular.tprocess.AdaptiveNotchFilter.mu` (the
step size) and :attr:`~acoular.tprocess.AdaptiveNotchFilter.gradient_leak`
(0 uses the instantaneous gradient, values towards 1 the full recursive one,
which tracks considerably better).

What the filter locked onto is available from
:attr:`~acoular.tprocess.AdaptiveNotchFilter.learned_frequencies`. It holds the
estimate for the block that was just yielded, so collect it while iterating to
get the full trajectory:

.. code-block:: python

    learned = []
    for block in auto.result(4096):
        learned.append(auto.learned_frequencies)
    learned = np.concatenate(learned)

:class:`~acoular.tprocess.CascadeNotchFilter` offers the same property, there
returning one trajectory per tonal source.

Autonomous tracking is convenient, but its limits are real: it needs a
dominant, well-separated tone with a limited rate of change, and the LMS
gradient gets weaker as the sampling rate rises. On a compact array where every
microphone sees a similar mix of several rotors, per-channel tracking has
little to lock onto. **If a reliable RPM signal exists, external mode is the
more robust choice.**

Harmonic series and multiple sources
------------------------------------

Propeller noise is a fundamental plus harmonics, and a multirotor has several
of them. :class:`~acoular.tprocess.CascadeNotchFilter` chains
:attr:`~acoular.tprocess.CascadeNotchFilter.num_sources` ×
:attr:`~acoular.tprocess.CascadeNotchFilter.harmonics_per_source` notches in
series and applies them to every channel. The
:attr:`~acoular.tprocess.CascadeNotchFilter.frequencies` array holds one row per
tonal source:

.. code-block:: python

    cascade = ac.CascadeNotchFilter(
        source=ts,
        num_sources=2,
        harmonics_per_source=3,
        frequencies=np.array([[200.0, 400.0, 600.0],
                              [150.0, 300.0, 450.0]]),
        pole_radius=0.999,
        zero_phase=True,
    )

Setting :attr:`~acoular.tprocess.CascadeNotchFilter.zero_phase` to ``True``
makes the children :class:`~acoular.tprocess.ZeroPhaseNotchFilter` instances, so
the whole cascade preserves phase.
:attr:`~acoular.tprocess.CascadeNotchFilter.pole_radius` accepts a scalar or a
per-stage array of shape ``(harmonics_per_source,)`` or
``(num_sources, harmonics_per_source)``, which is useful when higher harmonics
need a different width than the fundamental.

In external mode the cascade expects its
:attr:`~acoular.tprocess.CascadeNotchFilter.freq_source` to yield blocks of
shape ``(num, num_sources * harmonics_per_source)``, ordered source-major and
harmonic-minor -- one column per cascade stage. Note that this differs from
:class:`~acoular.tprocess.AdaptiveNotchFilter`, which expects a single column.

For auto mode, :attr:`~acoular.tprocess.CascadeNotchFilter.joint_lms` switches
between two strategies. With the default ``False`` every stage tracks
independently. Setting it to ``True`` selects the joint optimisation of
:cite:`Harvey2019`: one fundamental per source shared across all channels, with
harmonic *m* locked to *m* times that fundamental, and the gradients propagated
through the downstream stages. Harmonic locking is a strong constraint -- it
helps when the harmonics really are integer multiples, and hurts when they are
not.
