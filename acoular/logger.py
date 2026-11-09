# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Standard logging and optional live progress for Acoular.

Acoular objects expose a class-named :class:`logging.Logger` through ``self.logger``.
Common settings live on :data:`acoular.config`; application logging takes precedence
by default. Logging and progress belong in Python, not in compiled Numba kernels.

.. autosummary::
    :toctree: generated/

    progress
    track
"""

import logging
import sys
from contextlib import contextmanager, suppress
from importlib.util import find_spec
from os import environ
from threading import RLock

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
            _handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
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


class _LoggingMixin:
    """Provide class-level logging without adding Traits or instance state."""

    @property
    def logger(self):
        """The standard logger shared by instances of this concrete class.

        Ownership and default levels are refreshed on access. Saved logger references
        do not trigger another refresh. User subclasses remain under ``acoular`` so
        the same configuration and fallback apply to them.
        """
        cls = type(self)
        name = f'{cls.__module__}.{cls.__qualname__}'
        if not name.startswith('acoular.'):
            name = f'acoular.{name}'
        logger = logging.getLogger(name)
        _configure(logger)
        return logger


def _progress_enabled(console):
    setting = _settings().log_progress
    if setting == 'off':
        return False
    if console.is_jupyter and find_spec('ipywidgets') is None:
        return False
    if setting == 'on':
        return True
    if _has_application_handler(_namespace) or environ.get('CI'):
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

    def advance(self, amount=1):
        """Advance by completed work, not by work merely scheduled."""
        with _lock:
            if self._display is not None:
                self._display.advance(self._task_id, amount)
                if self._display.console.is_jupyter:
                    self._display.refresh()

    def update(self, *, completed=None, total=None):
        """Set completed work or a newly discovered total."""
        _check_total(total)
        with _lock:
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
    global _display  # noqa: PLW0603

    _check_total(total)
    _configure()
    with _lock:
        console = _get_console()
        if _progress_enabled(console):
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
            if display is not None:
                display.remove_task(task._task_id)
                if not display.tasks:
                    try:
                        display.stop()
                    finally:
                        _display = None


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
