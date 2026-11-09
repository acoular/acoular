# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Logging ownership, configuration and live display lifecycle."""

import io
import logging
import pickle
import subprocess
import sys
import warnings
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


@pytest.fixture
def calculation_setup(monkeypatch):
    monkeypatch.setattr(ac.config, 'global_caching', 'none')
    source = ac.TimeSamples(data=np.random.default_rng(7).normal(size=(512, 2)), sample_freq=8000)
    mics = ac.MicGeom(pos_total=np.array([[-0.1, 0.1], [0, 0], [0, 0]]))
    grid = ac.RectGrid(x_min=-0.1, x_max=0.1, y_min=0, y_max=0, z=1, increment=0.2)
    steer = ac.SteeringVector(mics=mics, grid=grid)
    spectra = ac.PowerSpectra(source=source, block_size=128, cached=False)
    return spectra, ac.BeamformerBase(freq_data=spectra, steer=steer, cached=False)


def test_csm_and_eigen_summaries_once(calculation_setup, caplog):
    caplog.set_level(logging.INFO, logger='acoular')
    spectra, _ = calculation_setup
    csm = spectra.csm
    values, vectors = spectra.eigvals, spectra.eigvecs
    assert csm.shape == (65, 2, 2)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(csm))
    np.testing.assert_allclose(csm @ vectors, vectors * values[:, None, :], atol=1e-12)
    assert sum(message.startswith('Computed cross-spectral matrices') for message in caplog.messages) == 1
    assert sum(message.startswith('Computed eigendecompositions') for message in caplog.messages) == 1
    assert len(caplog.records) == 2
    assert all(record.name == 'acoular.spectra.PowerSpectra' for record in caplog.records)


def test_nested_calculation_progress_cleanup(calculation_setup, caplog, mocker):
    caplog.set_level(logging.INFO, logger='acoular')
    ac.config.log_progress = 'on'
    counts = []
    original = al.Progress.add_task

    def add_task(display, *args, **kwargs):
        task_id = original(display, *args, **kwargs)
        counts.append(len(display.tasks))
        return task_id

    mocker.patch.object(al.Progress, 'add_task', add_task)
    _ = calculation_setup[0].eigvals
    assert max(counts) == 2
    assert al._display is None


def test_spectrum_cache_hit_has_no_info(calculation_setup, caplog, monkeypatch, tmp_path):
    spectra, _ = calculation_setup
    monkeypatch.setattr(ac.config, 'global_caching', 'all')
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    caplog.set_level(logging.DEBUG, logger='acoular')
    expected = spectra.csm[:]
    caplog.clear()
    other = ac.PowerSpectra(source=spectra.source, block_size=128)

    def unexpected_calculation(_self):
        pytest.fail('A warm cache must not calculate spectra again')

    monkeypatch.setattr(ac.PowerSpectra, 'calc_csm', unexpected_calculation)
    np.testing.assert_array_equal(other.csm, expected)
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert any(message.startswith('Reused cached spectra') for message in caplog.messages)


def test_beamforming_summarizes_original_batch(calculation_setup, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='acoular')
    _, beamformer = calculation_setup
    original = ac.BeamformerBase._calc
    calls = []

    def calculate(self, indices):
        calls.append(indices.copy())
        return original(self, indices)

    monkeypatch.setattr(ac.BeamformerBase, '_calc', calculate)
    result = beamformer.result[1:4].copy()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [1, 2, 3])
    summaries = [message for message in caplog.messages if message.startswith('Computed beamforming maps')]
    assert len(summaries) == 1 and 'frequencies=3' in summaries[0]
    caplog.clear()
    np.testing.assert_array_equal(beamformer.result[1:4], result)
    assert len(calls) == 1
    assert not caplog.records


@pytest.mark.parametrize('cls', [ac.BeamformerTime, ac.BeamformerTimeSq, ac.BeamformerCleant, ac.BeamformerCleantSq])
def test_time_beamforming_has_one_summary(cls, calculation_setup, caplog):
    caplog.set_level(logging.INFO, logger='acoular')
    spectra, frequency_beamformer = calculation_setup
    kwargs = {'n_iter': 1} if issubclass(cls, ac.BeamformerCleant) else {}
    beamformer = cls(source=spectra.source, steer=frequency_beamformer.steer, **kwargs)
    blocks = list(beamformer.result(64))
    assert blocks
    assert len(caplog.records) == 1
    message = caplog.messages[0]
    assert message.startswith('Computed time-domain beamforming')
    assert f'samples={sum(block.shape[0] for block in blocks)}' in message
    assert 'grid_points=2' in message


def test_time_beamforming_partial_summary(calculation_setup, caplog):
    caplog.set_level(logging.INFO, logger='acoular')
    spectra, frequency_beamformer = calculation_setup
    beamformer = ac.BeamformerTimeSq(source=spectra.source, steer=frequency_beamformer.steer)
    iterator = beamformer.result(64)
    block = next(iterator)
    iterator.close()
    assert len(caplog.records) == 1
    assert caplog.messages[0].startswith('Stopped time-domain beamforming')
    assert f'samples={len(block)}' in caplog.messages[0]
    assert 'reason=consumer_closed' in caplog.messages[0]


def test_csm_failure_has_no_success_or_error_log(calculation_setup, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='acoular')
    ac.config.log_progress = 'on'

    def fail(_self, _num):
        msg = 'source failed'
        raise RuntimeError(msg)

    monkeypatch.setattr(ac.TimeSamples, 'result', fail)
    with pytest.raises(RuntimeError, match='source failed'):
        _ = calculation_setup[0].csm
    assert not caplog.records
    assert al._display is None


@pytest.mark.usefixtures('setup_sounddevice_mocks')
def test_recording_does_not_query_properties_for_info(caplog, mocker):
    caplog.set_level(logging.INFO, logger='acoular')
    properties = mocker.patch.object(ac.SoundDeviceSamplesGenerator, 'device_properties')
    source = ac.SoundDeviceSamplesGenerator(num_samples=4, sample_freq=8000)
    list(source.result(4))
    properties.assert_not_called()


@pytest.mark.usefixtures('setup_sounddevice_mocks')
def test_recording_device_debug_fields(caplog):
    caplog.set_level(logging.DEBUG, logger='acoular')
    source = ac.SoundDeviceSamplesGenerator(num_samples=4, sample_freq=8000)
    list(source.result(4))
    selected = next(message for message in caplog.messages if message.startswith('Selected input device'))
    assert 'default_sample_freq=44100 Hz' in selected
    assert 'max_channels=2' in selected
    assert 'sample_freq=8000 Hz' in caplog.messages[-1]


@pytest.mark.usefixtures('setup_sounddevice_mocks')
def test_recording_overflow_warning_and_summary(caplog):
    import sounddevice as sd

    caplog.set_level(logging.INFO, logger='acoular')
    stream = sd.InputStream.return_value.__enter__.return_value
    stream.read.side_effect = [(np.ones((4, 1)), overflow) for overflow in [True, False, True]]
    source = ac.SoundDeviceSamplesGenerator(num_samples=10)
    with pytest.warns(UserWarning, match='Input overflow detected') as emitted:
        blocks = list(source.result(4))
    assert len(emitted) == 1
    assert [len(block) for block in blocks] == [4, 4, 2]
    assert not source.running
    assert len(caplog.records) == 2
    assert caplog.messages[0].startswith('Started recording')
    assert caplog.messages[-1].startswith('Stopped recording')
    assert 'overflow_reads=2' in caplog.messages[-1]
    assert 'samples=10' in caplog.messages[-1]
    stream.read.assert_called_with(4)


@pytest.mark.parametrize('warning_error', [False, True])
@pytest.mark.usefixtures('setup_sounddevice_mocks')
def test_recording_close_and_warning_error_cleanup(caplog, warning_error):
    import sounddevice as sd

    caplog.set_level(logging.INFO, logger='acoular')
    stream = sd.InputStream.return_value.__enter__.return_value
    stream.read.return_value = (np.ones((4, 1)), warning_error)
    source = ac.SoundDeviceSamplesGenerator(num_samples=10)
    iterator = source.result(4)
    if warning_error:
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(UserWarning, match='Input overflow'):
                next(iterator)
        assert len(caplog.records) == 1  # Start only; propagated failures aren't logged.
    else:
        next(iterator)
        iterator.close()
        assert 'samples=4' in caplog.messages[-1]
        assert 'reason=consumer_closed' in caplog.messages[-1]
    assert not source.running
    sd.InputStream.return_value.__exit__.assert_called_once()
    assert al._display is None


def test_data_file_download_and_local_resolution(tmp_path, monkeypatch, mocker, caplog):
    from acoular.tools.helpers import get_data_file

    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG, logger='acoular')
    connection = mocker.patch('http.client.HTTPSConnection').return_value
    response = connection.getresponse.return_value
    response.status = 200
    response.read.return_value = b'data'
    filename = get_data_file('logging_test.bin')
    assert filename.read_bytes() == b'data'
    infos = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert len(infos) == 1 and infos[0].startswith('Downloaded data file')
    assert 'bytes=4' in infos[0]
    connection.close.assert_called_once()
    caplog.clear()
    assert get_data_file('logging_test.bin') == filename
    assert all(record.levelno == logging.DEBUG for record in caplog.records)
    assert caplog.messages[0].startswith('Resolved data file')
    assert connection.request.call_count == 1


def test_download_failure_has_no_duplicate_log(tmp_path, monkeypatch, mocker, caplog):
    from acoular.tools.helpers import get_data_file

    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger='acoular')
    connection = mocker.patch('http.client.HTTPSConnection').return_value
    connection.getresponse.return_value.status = 404
    with pytest.raises(OSError, match='HTTP 404'):
        get_data_file('logging_missing.bin')
    connection.close.assert_called_once()
    assert not caplog.records
    assert not (tmp_path / 'logging_missing.bin').exists()
