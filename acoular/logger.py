# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Standard logging and optional live progress for Acoular.

Acoular objects expose a class-named :class:`logging.LoggerAdapter` through ``self.logger``.
Common settings live on :data:`acoular.config`; application logging takes precedence
by default. Logging and progress belong in Python, not in compiled Numba kernels.

.. autosummary::
    :toctree: generated/

    get_logger
    progress
    track
"""

import logging
import sys
from contextlib import contextmanager, suppress
from copy import copy
from functools import cache
from importlib.util import find_spec
from os import environ
from pathlib import Path
from threading import RLock
from time import perf_counter

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

_lock = RLock()
_namespace = logging.getLogger('acoular')
_managed_level = logging.NOTSET
_handler = None
_console = None
_console_color = None
_console_stream = None
_display = None


def _settings():
    # Config imports this module after the OpenBLAS check, before creating config.
    from .configuration import config  # noqa: PLC0415

    return config


def _has_application_handler(logger):
    """Check the propagation path, ignoring our fallback and null handlers."""
    while logger is not None:
        if any(handler is not _handler and not isinstance(handler, logging.NullHandler) for handler in logger.handlers):
            return True
        if not logger.propagate:
            break
        logger = logger.parent
    return False


def _get_console():
    global _console, _console_color, _console_stream  # noqa: PLW0603

    if _display is not None:
        return _display.console
    color = _settings().log_color
    if _console is None or _console_stream is not sys.stderr or _console_color != color:
        _console = Console(
            stderr=True,
            color_system=None if color == 'off' else 'auto',
            force_terminal=True if color == 'on' else None,
            no_color=False if color == 'on' else None,
        )
        _console_color = color
        _console_stream = sys.stderr
    return _console


def _fallback_filter(record):
    """Yield to application handlers, including ones installed after import."""
    if _has_application_handler(logging.getLogger(record.name)):
        return False
    _handler.console = _get_console()
    return True


class _ConsoleFormatter(logging.Formatter):
    """Shorten console labels without mutating records seen by other handlers."""

    def format(self, record):
        record = copy(record)
        record.name = record.name.rsplit('.', 1)[-1]
        return super().format(record)


def _configure(logger=None):
    """Refresh default ownership and levels at logger access or task entry."""
    global _handler, _managed_level  # noqa: PLW0603

    if logger is None:
        logger = _namespace
    with _lock:
        if _handler is None:
            _handler = RichHandler(
                console=_get_console(), show_time=False, show_path=False, markup=False, highlighter=NullHighlighter()
            )
            _handler.setFormatter(_ConsoleFormatter('%(name)s: %(message)s'))
            _handler.addFilter(_fallback_filter)
        if _handler not in _namespace.handlers:
            _namespace.addHandler(_handler)
        level = _settings().log_level
        if level != 'auto':
            level = logging.getLevelNamesMapping()[level]
        else:
            # Leave independently configured namespace levels alone. Class-specific
            # levels always retain ordinary logging semantics.
            if _namespace.level != _managed_level:
                return
            level = logging.NOTSET if _has_application_handler(logger) else logging.INFO
        if _namespace.level != level:
            _namespace.setLevel(level)
        _managed_level = level


def _format_fields(fields):
    """Render scalar context identically for every logging destination."""
    context = []
    for key, value in fields.items():
        if value is None:
            continue
        if key in (
            'elapsed',
            'sample_freq',
            'default_sample_freq',
            'output_sample_freq',
            'frequency',
            'selected_frequency',
            'lower_frequency',
            'upper_frequency',
        ):
            unit = 's' if key == 'elapsed' else 'Hz'
            value = format(value, '.3g' if key == 'elapsed' else 'g') + f' {unit}'
        elif isinstance(value, (str, Path)):
            value = str(value)
            if any(not char.isprintable() or char.isspace() or char in ',()' for char in value):
                value = repr(value)
        context.append(f'{key}={value}')
    return ', '.join(context)


class _LoggerAdapter(logging.LoggerAdapter):
    """Add flat formatted fields without owning logging configuration or output."""

    _CONFIG_ATTRIBUTES = frozenset({'level', 'handlers', 'filters', 'propagate', 'disabled', 'parent'})
    _LOG_KEYWORDS = frozenset({'exc_info', 'extra', 'stack_info', 'stacklevel'})

    def __init__(self, logger=None):
        super().__init__(_namespace if logger is None else logger, {})

    def __getattr__(self, name):
        # Pickle can inspect an adapter before its logger has been restored.
        logger = self.__dict__.get('logger')
        if logger is None:
            raise AttributeError(name)
        return getattr(logger, name)

    def __setattr__(self, name, value):
        if name in self._CONFIG_ATTRIBUTES:
            setattr(self.logger, name, value)
        else:
            super().__setattr__(name, value)

    def process(self, msg, kwargs):
        """Preserve explicit standard extra attributes, including on Python 3.11."""
        if self.extra:
            kwargs['extra'] = {**self.extra, **(kwargs.get('extra') or {})}
        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        """Accept scalar fields alongside the standard logging keyword arguments."""
        if self.isEnabledFor(level):
            fields = {key: kwargs.pop(key) for key in tuple(kwargs) if key not in self._LOG_KEYWORDS}
            context = _format_fields(fields)
            if context:
                # Fields are literal text, not part of the caller's %-template.
                if args:
                    context = context.replace('%', '%%')
                msg = f'{msg} ({context})'
            kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
            super().log(level, msg, *args, **kwargs)

    def operation(self, description, completed, *, total=None, count=None, stopped=None, **fields):
        """Time work and report its outcome; update late metadata via task.fields."""
        return _operation(self, description, completed, total=total, count=count, stopped=stopped, fields=fields)


@cache
def _get_adapter(logger):
    return _LoggerAdapter(logger)


def get_logger(name='acoular'):
    """Return a shared formatting adapter around the standard named logger.

    Parameters
    ----------
    name : str, optional
        Logger name. Acoular modules use their full ``acoular.*`` name.

    Returns
    -------
    logging.LoggerAdapter
        Adapter accepting flat context keywords on ``debug``, ``info``, etc.,
        and providing ``operation(...)``. Its ``logger`` attribute is exactly
        ``logging.getLogger(name)``. Configuration remains on that standard logger.
        Ordinary logging keywords retain their normal meaning.
    """
    logger = logging.getLogger(name)
    _configure(logger)
    with _lock:
        return _get_adapter(logger)


def _progress_enabled(console, logger=None):
    if logger is None:
        logger = _namespace
    setting = _settings().log_progress
    if setting == 'off':
        return False
    if console.is_jupyter and find_spec('ipywidgets') is None:
        return False
    if setting == 'on':
        return True
    if _has_application_handler(logger) or environ.get('CI'):
        return False
    if console.is_jupyter:
        return True
    return sys.stderr.isatty() and not console.is_dumb_terminal


def _check_total(total):
    if total is not None and total < 0:
        msg = 'Progress total must be non-negative or None.'
        raise ValueError(msg)


class _ProgressTask:
    """A task handle which is also usable when display is disabled."""

    def __init__(self, display=None, task_id=None):
        self._display = display
        self._task_id = task_id
        self.completed = 0
        self._closed = False

    def advance(self, amount=1):
        """Advance by completed work, not by work merely scheduled."""
        with _lock:
            if self._closed:
                return
            self.completed += amount
            if self._display is not None:
                self._display.advance(self._task_id, amount)
                if self._display.console.is_jupyter:
                    self._display.refresh()

    def update(self, *, completed=None, total=None):
        """Set completed work or a newly discovered total."""
        _check_total(total)
        with _lock:
            if self._closed:
                return
            if completed is not None:
                self.completed = completed
            if self._display is not None:
                self._display.update(self._task_id, completed=completed, total=total, refresh=True)


@contextmanager
def progress(description, total=None):
    """Display a task, or silently do the same work when progress is disabled.

    Parameters
    ----------
    description : str
        Label of the processing task.
    total : float or None, optional
        Total units of work. Without a known total, a spinner indicates activity.

    Yields
    ------
    task
        Handle with ``advance(amount=1)`` and ``update(completed=..., total=...)``.
        Nested and overlapping contexts share a display. On exit the task is removed,
        without marking interrupted work as complete or suppressing exceptions.

    Notes
    -----
    Automatic display requires an interactive terminal or a notebook with widget
    support, and no application logging handlers. Explicitly enabling progress
    bypasses automatic ownership/detection, but still requires notebook widgets.
    Existing tasks keep their display policy until exit. stdout/stderr are never
    redirected. This helper must not be called from compiled Numba functions.
    """
    with _progress(description, total, _namespace) as task:
        yield task


@contextmanager
def _progress(description, total, logger):
    """Share rendering while respecting the initiating logger's handler route."""
    global _display  # noqa: PLW0603

    _check_total(total)
    _configure(logger)
    with _lock:
        console = _get_console()
        if _progress_enabled(console, logger):
            if _display is None:
                display = Progress(
                    SpinnerColumn(),
                    TextColumn('{task.description}', markup=False),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    auto_refresh=not console.is_jupyter,
                    transient=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                display.start()
                _display = display
            task = _ProgressTask(_display, _display.add_task(description, total=total))
        else:
            task = _ProgressTask()
    try:
        yield task
    finally:
        with _lock:
            display = task._display
            task._display = None
            task._closed = True
            if display is not None:
                display.remove_task(task._task_id)
                if not display.tasks:
                    try:
                        display.stop()
                    finally:
                        _display = None


@contextmanager
def _operation(logger, description, completed, *, total=None, count=None, stopped=None, fields=None):
    """Time selected work; report success or intentional closure, never exceptions.

    Fields may be populated as work discovers metadata. A reason field indicates
    an intentional stop. Counts track actual work even with display disabled.
    """
    fields = {} if fields is None else fields
    _configure(logger.logger)
    start = perf_counter()
    logger.debug(description, **fields)
    with _progress(description, total, logger.logger) as task:
        task.fields = fields

        def summarize(message, reason=None):
            context = fields.copy()
            if count is not None:
                context[count] = task.completed
            if reason is not None:
                context['reason'] = reason
            context['elapsed'] = perf_counter() - start
            logger.info(message, **context)

        try:
            yield task
        except GeneratorExit:
            if stopped is not None:
                summarize(stopped, reason='consumer_closed')
            raise
        else:
            summarize(stopped if stopped is not None and fields.get('reason') else completed)


def track(iterable, description='Processing', total=None):
    """Yield items while reporting completed iterations.

    Parameters
    ----------
    iterable : iterable
        Items to process. The iterable's length is used when available.
    description : str, optional
        Label of the processing task.
    total : float or None, optional
        Override the inferred total. Unknown-length iterables use a spinner.

    Yields
    ------
    object
        Each item from the original iterable, unchanged. Progress advances when
        iteration resumes after the caller processes an item.

    Notes
    -----
    If retaining this iterator after an early exit, close it to release its display,
    or use :func:`progress` for explicit context-managed task lifetime.
    """
    if total is None:
        with suppress(TypeError):
            total = len(iterable)
    with progress(description, total=total) as task:
        for item in iterable:
            yield item
            task.advance()
