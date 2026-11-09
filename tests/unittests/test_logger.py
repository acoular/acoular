# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Logging ownership, configuration and live display lifecycle."""

import io
import logging
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import acoular as ac
from acoular import logger as al

from tests.utils import create_instance, get_all_classes

import numpy as np
import pytest
from traits.trait_errors import TraitError


@pytest.fixture(autouse=True)
def logging_state(monkeypatch):
    """Isolate logging without closing or replacing pytest's own handlers."""
    root = logging.getLogger()
    namespace = logging.getLogger('acoular')
    saved = {}
    for log in [root, namespace, *logging.Logger.manager.loggerDict.values()]:
        if isinstance(log, logging.Logger) and (log is root or log.name.startswith('acoular')):
            saved[log] = (log.handlers[:], log.level, log.propagate, log.disabled)
    settings = ac.config.trait_get('log_level', 'log_color', 'log_progress')
    ac.config.trait_setq(log_level='auto', log_color='off', log_progress='off')
    for log in saved:
        log.handlers = []
        log.setLevel(logging.WARNING if log is root else logging.NOTSET)
        log.propagate = True
        log.disabled = False
    monkeypatch.setattr(al, '_managed_level', logging.NOTSET)
    monkeypatch.setattr(al, '_handler', None)
    monkeypatch.setattr(al, '_console', None)
    monkeypatch.setattr(al, '_console_color', None)
    monkeypatch.setattr(al, '_console_stream', None)
    try:
        yield
    finally:
        if al._display is not None:
            al._display.stop()
            al._display = None
        ac.config.trait_setq(**settings)
        for log in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(log, logging.Logger) and log.name.startswith('acoular') and log not in saved:
                log.handlers = []
                log.setLevel(logging.NOTSET)
                log.propagate = True
                log.disabled = False
        for log, (handlers, level, propagate, disabled) in saved.items():
            log.handlers = handlers
            log.setLevel(level)
            log.propagate = propagate
            log.disabled = disabled


def application_logging(level=logging.WARNING, handler_level=logging.NOTSET, name=''):
    logging.getLogger().handlers.clear()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(handler_level)
    log = logging.getLogger(name)
    log.addHandler(handler)
    log.setLevel(level)
    return stream


def test_standalone_output(capsys):
    logging.getLogger().handlers.clear()  # pytest installs capture handlers after fixture setup.
    obj = ac.WNoiseGenerator()
    obj.logger.debug('not visible')
    obj.logger.info('plain [brackets]')
    out, err = capsys.readouterr()
    assert out == ''
    assert 'plain [brackets]' in err
    assert 'not visible' not in err
    assert 'WNoiseGenerator:' in err
    assert 'acoular.signals.' not in err
    assert '\x1b[' not in err
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger().handlers == []


@pytest.mark.parametrize('late', [False, True])
def test_application_levels_and_no_duplicate_output(late, capsys):
    obj = ac.WNoiseGenerator()
    if late:
        obj.logger.info('before setup')
        capsys.readouterr()
    stream = application_logging()
    obj.logger.info('suppressed')
    obj.logger.warning('one warning')
    assert stream.getvalue() == 'one warning\n'
    assert capsys.readouterr() == ('', '')


def test_handler_threshold_does_not_activate_fallback(capsys):
    stream = application_logging(logging.DEBUG, logging.ERROR)
    ac.WNoiseGenerator().logger.warning('filtered by application')
    assert stream.getvalue() == ''
    assert capsys.readouterr() == ('', '')


def test_null_handler_does_not_take_ownership(capsys):
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())
    ac.WNoiseGenerator().logger.info('still standalone')
    assert 'still standalone' in capsys.readouterr().err


def test_namespace_handler_and_propagation(capsys):
    stream = application_logging(logging.ERROR, name='acoular.signals')
    logging.getLogger('acoular.signals').propagate = False
    ac.WNoiseGenerator().logger.warning('suppressed')
    ac.WNoiseGenerator().logger.error('application error')
    assert stream.getvalue() == 'application error\n'
    assert capsys.readouterr() == ('', '')


def test_propagation_boundary_ignores_handlers_above_it():
    application_logging()
    log = logging.getLogger('acoular.signals')
    log.propagate = False
    assert not al._has_application_handler(log)


def test_saved_logger_refresh_boundary(capsys):
    logging.getLogger().handlers.clear()
    obj = ac.WNoiseGenerator()
    saved = obj.logger
    stream = application_logging()
    saved.info('before refresh')
    assert stream.getvalue() == 'before refresh\n'
    assert obj.logger is saved
    saved.info('after refresh')
    assert stream.getvalue() == 'before refresh\n'
    assert capsys.readouterr() == ('', '')


def test_explicit_level_and_return_to_auto():
    stream = application_logging()
    obj = ac.WNoiseGenerator()
    saved = obj.logger
    ac.config.log_level = 'DEBUG'
    saved.debug('explicit debug')
    ac.config.log_level = 'auto'
    saved.info('hidden again')
    assert stream.getvalue() == 'explicit debug\n'


def test_standard_class_and_namespace_overrides(capsys):
    logging.getLogger().handlers.clear()
    logging.getLogger('acoular').setLevel(logging.ERROR)
    obj = ac.WNoiseGenerator()
    assert obj.logger.getEffectiveLevel() == logging.ERROR
    obj.logger.setLevel(logging.DEBUG)
    obj.logger.debug('class override')
    assert 'class override' in capsys.readouterr().err


def test_handler_is_not_duplicated():
    for _ in range(3):
        _ = ac.WNoiseGenerator().logger
    handlers = logging.getLogger('acoular').handlers
    assert len(handlers) == 1
    assert handlers[0] is al._handler


def test_color_override(capsys):
    logging.getLogger().handlers.clear()
    ac.config.log_color = 'on'
    ac.WNoiseGenerator().logger.warning('colored')
    assert '\x1b[' in capsys.readouterr().err
    ac.config.log_color = 'off'
    ac.WNoiseGenerator().logger.warning('uncolored')
    assert '\x1b[' not in capsys.readouterr().err


@pytest.mark.parametrize('setting', ['log_level', 'log_color', 'log_progress'])
def test_settings_validate(setting):
    with pytest.raises(TraitError):
        setattr(ac.config, setting, 'invalid')


def test_independent_config_does_not_reconfigure_logging():
    from acoular.configuration import Config

    logging.getLogger().handlers.clear()
    _ = ac.WNoiseGenerator().logger
    Config().log_level = 'DEBUG'
    assert logging.getLogger('acoular').level == logging.INFO


@pytest.mark.parametrize('cls', get_all_classes(hastraits_only=True))
def test_class_logger_without_instance_or_trait_state(cls):
    obj = create_instance(cls)
    before = obj.__dict__.copy()
    log = obj.logger
    assert isinstance(log, logging.Logger)
    assert log.name == f'{cls.__module__}.{cls.__qualname__}'
    assert obj.logger is log
    assert 'logger' not in obj.traits()
    assert obj.__dict__ == before


def test_shared_class_loggers_and_user_subclasses():
    class UserNoise(ac.WNoiseGenerator):
        pass

    assert ac.WNoiseGenerator().logger is ac.WNoiseGenerator().logger
    assert UserNoise().logger is not ac.WNoiseGenerator().logger
    assert UserNoise().logger.name.startswith('acoular.')
    assert UserNoise().logger.name.endswith('.UserNoise')


def test_pickle_after_logger_access():
    obj = ac.WNoiseGenerator(seed=42)
    _ = obj.logger
    restored = pickle.loads(pickle.dumps(obj))
    assert restored.seed == obj.seed
    assert restored.digest == obj.digest
    assert restored.logger is obj.logger


def test_disabled_tasks_and_iteration():
    with al.progress('Disabled') as task:
        task.advance()
        task.update(completed=2, total=10)
        assert al._display is None
    assert list(al.track(range(3))) == [0, 1, 2]


def test_nested_tasks_and_unmodified_streams():
    ac.config.log_progress = 'on'
    stdout, stderr = sys.stdout, sys.stderr
    with al.progress('Outer', total=10) as outer:
        display = al._display
        outer.advance(2)
        with al.progress('Inner') as inner:
            assert al._display is display
            assert len(display.tasks) == 2
            assert display.tasks[1].total is None
            inner.update(completed=3, total=5)
            assert display.tasks[1].completed == 3
            assert display.tasks[1].total == 5
            assert sys.stdout is stdout and sys.stderr is stderr
        assert len(display.tasks) == 1
        assert display.tasks[0].completed == 2
        assert display.tasks[0].total == 10
    assert al._display is None
    outer.advance()  # A retained handle becomes inactive after its context closes.
    outer.update(completed=99)
    assert outer.completed == 2


def test_live_logs_share_the_display_console(monkeypatch):
    logging.getLogger().handlers.clear()
    stream = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', stream)
    ac.config.log_color = 'on'
    ac.config.log_progress = 'on'
    with al.progress('Active', total=2):
        display = al._display
        ac.WNoiseGenerator().logger.info('a log [literal]')
        assert al._handler.console is display.console
        assert sys.stderr is stream
    assert stream.getvalue().count('a log [literal]') == 1
    assert '\x1b[?25l' in stream.getvalue()  # Cursor hidden while live.
    assert '\x1b[?25h' in stream.getvalue()  # Cursor restored on exit.


def test_application_progress_handover_and_explicit_override():
    ac.config.log_progress = 'auto'
    application_logging()
    with al.progress('Application owned'):
        assert al._display is None
    ac.config.log_progress = 'on'
    with al.progress('Explicit display'):
        assert al._display is not None
    assert al._display is None


def test_progress_cleanup_on_exception():
    ac.config.log_progress = 'on'
    with pytest.raises(RuntimeError, match='computation failed'), al.progress('Task', total=10) as task:
        display = al._display
        task.advance(2)
        failed_task = display.tasks[0]
        msg = 'computation failed'
        raise RuntimeError(msg)
    assert al._display is None
    assert failed_task.completed == 2
    assert not failed_task.finished
    with al.progress('Restart'):
        assert al._display is not None
    assert al._display is None


def test_threaded_task_updates():
    ac.config.log_progress = 'on'
    with al.progress('Threaded', total=100) as task, ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: task.advance(), range(100)))
        assert al._display.tasks[0].completed == 100


def test_track_inference_completion_and_early_close():
    ac.config.log_progress = 'on'
    iterator = al.track([1, 2, 3])
    assert next(iterator) == 1
    task = al._display.tasks[0]
    assert task.total == 3 and task.completed == 0
    assert next(iterator) == 2
    assert task.completed == 1
    iterator.close()
    assert al._display is None
    iterator = al.track(iter([1, 2]))
    assert next(iterator) == 1
    assert al._display.tasks[0].total is None
    assert list(iterator) == [2]
    assert al._display is None
    assert list(al.track([])) == []
    assert al._display is None


@pytest.mark.parametrize('setting', ['on', 'off'])
def test_negative_total_rejected(setting):
    ac.config.log_progress = setting
    with pytest.raises(ValueError, match='non-negative'), al.progress('Invalid', total=-1):
        pass
    with al.progress('Task') as task, pytest.raises(ValueError, match='non-negative'):
        task.update(total=-1)
    assert al._display is None


def test_automatic_progress_policy(monkeypatch):
    logging.getLogger().handlers.clear()
    ac.config.log_progress = 'auto'
    console = al._get_console()
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    monkeypatch.setenv('TERM', 'xterm-256color')
    assert al._progress_enabled(console)
    monkeypatch.setenv('CI', 'true')
    assert not al._progress_enabled(console)
    monkeypatch.delenv('CI')
    application_logging()
    assert not al._progress_enabled(console)
    ac.config.log_progress = 'on'
    assert al._progress_enabled(console)


def test_forced_color_does_not_enable_redirected_progress():
    ac.config.log_color = 'on'
    ac.config.log_progress = 'auto'
    assert not al._progress_enabled(al._get_console())


def test_notebook_policy_without_install_warnings(monkeypatch):
    ac.config.log_progress = 'auto'
    console = al._get_console()
    monkeypatch.setattr(console, 'is_jupyter', True)
    monkeypatch.setattr(al, 'find_spec', lambda _: None)
    assert not al._progress_enabled(console)
    ac.config.log_progress = 'on'
    assert not al._progress_enabled(console)
    monkeypatch.setattr(al, 'find_spec', lambda _: object())
    assert al._progress_enabled(console)
    ac.config.log_progress = 'auto'
    logging.getLogger().handlers.clear()
    monkeypatch.delenv('CI', raising=False)
    assert al._progress_enabled(console)


def test_notebook_tasks_refresh_explicitly(monkeypatch, mocker):
    ac.config.log_progress = 'on'
    console = al._get_console()
    monkeypatch.setattr(console, 'is_jupyter', True)
    monkeypatch.setattr(al, 'find_spec', lambda _: object())
    # Exercise the notebook update policy without requiring a real widget frontend.
    mocker.patch('rich.live.Live.refresh')
    refresh = mocker.patch.object(al.Progress, 'refresh')
    with al.progress('Notebook', total=2) as task:
        assert not al._display.live.auto_refresh
        task.advance()
        task.update(completed=2)
        assert refresh.call_count >= 2
    assert al._display is None


def test_import_leaves_root_logging_untouched():
    code = """
import logging
root = logging.getLogger()
handler = logging.NullHandler()
root.addHandler(handler)
root.setLevel(logging.ERROR)
logger_class = logging.getLoggerClass()
import acoular
assert root.handlers == [handler]
assert root.level == logging.ERROR
assert logging.getLoggerClass() is logger_class
assert not logging.getLogger('acoular').handlers
assert not logging.getLogger('acoular').level
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ''


def test_console_formatter_preserves_record_names():
    record = logging.LogRecord('acoular.spectra.PowerSpectra', logging.INFO, __file__, 1, 'message', (), None)
    formatter = al._ConsoleFormatter('%(name)s: %(message)s')
    assert formatter.format(record) == 'PowerSpectra: message'
    assert record.name == 'acoular.spectra.PowerSpectra'


def test_event_fields_and_disabled_formatting(caplog):
    caplog.set_level(logging.INFO, logger='acoular')
    logger = al._get_logger('acoular.tools.helpers')
    al._event(
        logger, logging.INFO, 'Wrote file', file='folder/file name.h5', sample_freq=51200, elapsed=0.125, optional=None
    )
    assert caplog.messages == ["Wrote file (file='folder/file name.h5', sample_freq=51200 Hz, elapsed=0.125 s)"]

    class Unformattable:
        def __str__(self):
            msg = 'Disabled messages must not format fields'
            raise AssertionError(msg)

    al._event(logger, logging.DEBUG, 'Not emitted', value=Unformattable())
    assert len(caplog.records) == 1


def test_operation_respects_class_handler_ownership(monkeypatch):
    logging.getLogger().handlers.clear()
    log = logging.getLogger('acoular.signals.WNoiseGenerator')
    stream = io.StringIO()
    log.addHandler(logging.StreamHandler(stream))
    ac.config.log_progress = 'auto'
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    monkeypatch.setenv('TERM', 'xterm-256color')
    with al._operation(ac.WNoiseGenerator().logger, 'Processing', 'Processed'):
        assert al._display is None
    assert log.getEffectiveLevel() == logging.WARNING
    assert stream.getvalue() == ''


def test_operation_counts_and_timing(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='acoular')
    monkeypatch.setattr(al, 'perf_counter', iter([10.0, 12.5]).__next__)
    with al._operation(al._get_logger('acoular.test'), 'Processing', 'Processed', count='samples') as task:
        task.advance(3)
        task.update(completed=5)
    assert caplog.messages == ['Processed (samples=5, elapsed=2.5 s)']


def test_operation_failures_are_not_logged(caplog):
    caplog.set_level(logging.INFO, logger='acoular')
    ac.config.log_progress = 'on'
    with (
        pytest.raises(RuntimeError, match='failed'),
        al._operation(al._get_logger('acoular.test'), 'Processing', 'Processed'),
    ):
        msg = 'failed'
        raise RuntimeError(msg)
    assert caplog.messages == []
    assert al._display is None


def test_pink_depth_warning_preserves_signal(caplog):
    requested = ac.PNoiseGenerator(num_samples=64, depth=16, seed=7)
    supported = ac.PNoiseGenerator(num_samples=64, depth=6, seed=7)
    with pytest.warns(UserWarning, match='requested_depth=16, depth=6, samples=64') as emitted:
        actual = requested.signal()
    assert len(emitted) == 1
    np.testing.assert_array_equal(actual, supported.signal())
    assert requested.depth == 16
    assert not caplog.records


def test_invalid_band_width_raises_without_logging(caplog):
    with pytest.raises(ValueError, match='num must be 1 .* or 3 .* got 2'):
        ac.tools.barspectrum(np.ones(3), np.arange(3), num=2)
    assert not caplog.records
