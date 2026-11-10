Logging and progress
====================

Acoular provides standard Python logging with Rich console output and optional
progress displays. Importing Acoular does not modify the root logger or install
console handlers. The default output is initialized when an object's ``logger``
attribute is accessed, :func:`acoular.logger.get_logger` is called, or a progress
task starts. No user setup is required.

Selected computations and I/O operations emit concise summaries and automatically
use progress displays when appropriate. Routine processing blocks remain quiet at
INFO. Existing Python warnings retain their filtering and exception semantics.

Common settings
---------------

Three settings on :data:`acoular.config` control the built-in behavior. The
snippets on this page come from the runnable
:doc:`../auto_examples/introductory_examples/example_logging` example.

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Import packages.
   :end-before: # %%

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Configure common settings.
   :end-before: # %%

``log_level`` accepts ``'auto'``, ``'DEBUG'``, ``'INFO'``, ``'WARNING'``,
``'ERROR'``, ``'CRITICAL'`` and ``'NOTSET'``. Both ``log_color`` and
``log_progress`` accept ``'auto'``, ``'on'`` and ``'off'``.

All three default to ``'auto'``. With no application logging handlers, the
logging threshold is INFO. Color follows Rich's environment and terminal
detection, including ``NO_COLOR``. Automatic progress requires an interactive,
non-dumb terminal or a notebook with widget support; it is disabled in CI and
when application logging owns output. Explicit ``'on'`` requests progress even
when application logging is present, but Rich still determines what the output
device can render.

Acoular-owned output goes to stderr, leaving stdout available for results.
Neither logging nor progress redirects ``sys.stdout`` or ``sys.stderr``.
Unrelated print statements may therefore disturb a live progress display.

Application logging
-------------------

Acoular loggers live under the ``acoular`` namespace. If an application installs
a handler along a logger's propagation path, Acoular's built-in handler steps
aside, even if the application filters the message out. Null handlers do not
count as application output handlers. No application handlers are removed or
reconfigured.

For example, standard application configuration works before or after importing
Acoular:

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Integrate with application logging, even after Acoular was imported.
   :end-before: # %%

With ``log_level='auto'``, Acoular refreshes its default threshold on access to
``self.logger`` or entry to a progress task. A saved logger reference does not
itself refresh the threshold, and an already-running task keeps its display
policy until exit. This is not continuous monitoring of application logging.
An explicit ``ac.config.log_level`` updates the Acoular namespace immediately
and overrides automatic level handover. Application handler thresholds still
apply normally.

Class-specific levels retain normal Python logging behavior. File handlers,
rotating files, custom formatters, filters and destinations should be configured
with Python's :mod:`logging` module, rather than Acoular-specific settings.
Python warnings and Numba's own diagnostic loggers remain independent.

Object loggers
--------------

Acoular's Traits-based objects expose a small :class:`logging.LoggerAdapter`
through ``self.logger``. Instances of the same concrete class share that adapter;
accessing it adds no instance state or Traits and does not change computation
digests. A built-in class such as ``WNoiseGenerator`` uses the logger name
``acoular.signals.WNoiseGenerator``.

The usual ``debug``, ``info``, ``warning``, ``error`` and ``exception`` methods
accept flat scalar context keywords. Common fields such as ``elapsed`` and
``sample_freq`` receive their units automatically. Context is formatted only when
the logger's level is enabled and is included in the ordinary message before
handler dispatch, not just by Rich. An application's ``%(message)s`` formatter
therefore receives the complete message, with no custom message objects required.
Positional ``%`` formatting and the standard ``extra``, ``exc_info``, ``stack_info`` and
``stacklevel`` keywords retain their normal meanings; these keywords are not
context fields.

Module-level code needs only the logger factory:

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Obtain a module-level logger with flat context fields.
   :end-before: # %%

The adapter has no independent handlers or logging threshold. Its ``logger``
attribute is exactly the object returned by ``logging.getLogger(name)``.
Application configuration of that named logger, its ancestors, or the root logger
works unchanged. Standard configuration methods such as ``setLevel``,
``addHandler`` and ``removeHandler`` are delegated, as are configuration
attributes including ``handlers``, ``filters``, ``propagate`` and ``disabled``.
For APIs requiring a concrete :class:`logging.Logger`, use ``obj.logger.logger``.

User subclasses also receive class-named loggers. Classes outside the Acoular
package have their module and qualified class name prefixed with ``acoular.``
so that Acoular's configuration also applies to them. Registered logger objects
and log levels remain standard; only Acoular's adapter adds the convenience API.

Message policy
--------------

The default console shows the class name (or final module component), followed by
an action and scalar context fields. Full logger names remain in the underlying
records. No wall-clock timestamp is shown; operation summaries include duration::

    INFO PowerSpectra: Computed eigendecompositions (frequencies=65, channels=2, elapsed=0.012 s)

Levels have the following meanings:

* **DEBUG**: cache decisions, configuration, PSF calculations and useful execution
  details. These messages do not dump arrays or complete object representations.
* **INFO**: concise summaries of selected major computations and user-facing I/O.
  Recording starts are also reported because acquisition may run indefinitely.
* **WARNING**: recoverable operational anomalies where a Python warning is not
  the appropriate API. Do not report the same event through both mechanisms.
* **ERROR**: a failure handled by Acoular without propagating an exception.
  Propagated exceptions are not also logged; their messages provide context.
* **CRITICAL**: no planned use in ordinary library code.

Automatic INFO summaries cover CSM computation, batch eigendecomposition,
frequency-domain beamforming requests, time-domain beamforming runs, recordings,
file exports and downloads. Nested major computations produce separate summaries;
PSFs, individual frequencies and routine transforms do not produce INFO chatter.
Cached results do not announce a new computation. Local data-file resolution is
DEBUG; an actual successful download is INFO. Solver implementations and their
explicit third-party output controls are not instrumented.

``elapsed`` is inclusive wall-clock time, measured with a monotonic clock. It
includes nested work, input reads and first-call compilation during the operation,
not just numerical kernel time. Streaming durations include time spent suspended
between yields: they measure session lifetime, not isolated execution time.
Nested durations are not additive. Progress and summaries count actual work
without splitting numerical batches or evaluating lazy results merely to populate
messages. Counts and metadata are included when
available; a frequency-domain request may use an indeterminate display because
its solver exposes no intermediate updates.

Normal completion and intentional early closure are distinct. Selected streaming
operations report a partial INFO summary on explicit generator closure, including
actual counts and ``reason=consumer_closed``. Explicit stop flags use
``reason=requested_stop``. Exceptions do not produce false success summaries or
duplicate ERROR records. Export files and acquisition streams are closed before
their stop summaries are emitted.

Python warnings keep their normal presentation and filtering; Acoular does not
install process-wide warning capture. New data-quality warnings cover pink-noise
depth reduction and the first audio-input overflow per recording. The final
recording summary includes ``overflow_reads``, not an invented lost-sample count.
These warnings can intentionally become exceptions under ``-Werror``. The
``overflow`` trait continues to describe the last read. Unsupported
``barspectrum(..., num=...)`` widths now raise ``ValueError`` instead of printing
and returning a sentinel tuple.

ASCII demo maps remain result output on stdout. The demo's optional Matplotlib
fallback notice is INFO. Developer-only diagnostic output remains explicit,
and no third-party stdout/stderr streams are captured.

Progress tasks
--------------

Use ``self.logger.operation(...)`` to combine a progress task with a timed
summary. Pass context fields directly rather than importing lifecycle helpers:


.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Process samples with a known total.
   :end-before: # %%

The operation's ``count`` names the actual-work counter in its summary. Additional
metadata discovered during processing can be added with ``task.fields.update(...)``.
Normal completion emits the supplied outcome message. The optional ``stopped``
message distinguishes intentional early closure; exceptions are propagated
without a success record.

For a display without an automatic log summary, the standalone
:func:`acoular.logger.progress` helper remains available. Use ``total=None``
(the default) when the amount of work is unknown. For example,
indefinite sources must not pass their ``num_samples=-1`` sentinel as a total.
An unknown-duration task can wrap sample generation:

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Generate samples with unknown-duration activity.
   :end-before: # %%

Unknown-total tasks show activity rather than a percentage. A task can discover
its total later with ``task.update(total=100)`` or set its completed count with
``task.update(completed=25)``. Task totals must be non-negative.

For simple iteration, :func:`acoular.logger.track` infers a total when the
iterable has a length:

.. literalinclude:: ../../examples/introductory_examples/example_logging.py
   :language: python
   :start-after: # Track an iterable and emit class-level messages.
   :end-before: # %%

Progress advances after the caller processes each item. If retaining a tracking
iterator after an early exit, close it to release its display, or use the explicit
``progress`` context manager instead.

Nested and overlapping tasks share one Rich display, coordinated with Acoular's
own log messages. Updates from Python threads are supported; there is no
cross-process progress aggregation. Disabled tasks have the same update API but
produce no display. Context exit removes tasks and cleans up the display without
suppressing exceptions or falsely completing failed work.

Notebook support is best effort. Live notebook displays require ``ipywidgets``;
when it is unavailable, progress is disabled without installation warnings.
Rich displays are refreshed explicitly in notebooks. Widget installation alone
does not guarantee compatibility with every notebook frontend.

Compiled numerical functions
----------------------------

Log messages and update progress only on the Python side of Numba calls. Do not
pass logger or task objects into compiled kernels or introduce object-mode
callbacks for display updates. Update progress between calls, using units such
as completed frequencies or processed samples. For a single long call, use an
unknown-total activity task rather than inventing a percentage.

An activity indicator may pause during first-call compilation or a kernel which
holds the GIL. Animation is not guaranteed during these calls. The logging
infrastructure does not change kernel signatures, compilation settings or
numerical algorithms.
