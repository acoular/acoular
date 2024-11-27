import acoular as ac
import pytest


def test_load_mics(aiaa_bechmark_time_data_file):
    """Test if microphone positions are loaded correctly from an AIAA benchmark file.

    Test for deprecation warning when using from_file attribute

    Parameters
    ----------
    aiaa_bechmark_time_data_file : str
        Path to the AIAA benchmark file with time data (fixture).

    """
    with pytest.deprecated_call():
        mic_geom = ac.tools.MicAIAABenchmark(from_file=aiaa_bechmark_time_data_file)
    mic_geom = ac.tools.MicAIAABenchmark(file=aiaa_bechmark_time_data_file)
    assert mic_geom.num_mics == 8
    assert mic_geom.mpos.shape == (3, 8)
    assert mic_geom.mpos_tot.shape == (3, 8)


def test_load_csm(aiaa_bechmark_csm_file):
    """Test if cross spectral matrix is loaded correctly from an AIAA benchmark file.

    Test for deprecation warning when using from_file attribute

    Parameters
    ----------
    aiaa_bechmark_csm_file : str
        Path to the AIAA benchmark file with cross spectral matrix data (fixture
    """
    with pytest.deprecated_call():
        csm = ac.tools.CsmAIAABenchmark(name=aiaa_bechmark_csm_file)
    csm = ac.tools.CsmAIAABenchmark(file=aiaa_bechmark_csm_file)
    assert csm.numchannels == 8
    assert csm.csm.shape == (csm.fftfreq().shape[0], csm.numchannels, csm.numchannels)


def test_load_time(aiaa_bechmark_time_data_file):
    """Test if time data is loaded correctly from an AIAA benchmark file.

    Test for deprecation warning when using from_file attribute

    Parameters
    ----------
    aiaa_bechmark_time_data_file : str
        Path to the AIAA benchmark file with time data (fixture).
    """
    with pytest.deprecated_call():
        time_data = ac.tools.TimeSamplesAIAABenchmark(name=aiaa_bechmark_time_data_file)
    time_data = ac.tools.TimeSamplesAIAABenchmark(file=aiaa_bechmark_time_data_file)
    assert time_data.numchannels == 8
    assert time_data.numsamples == 480
    assert time_data.sample_freq == 48000
    assert time_data.data.shape == (time_data.numsamples, time_data.numchannels)
    assert time_data.metadata != {}


def test_load_trigger(aiaa_bechmark_trigger_file):
    """Test if trigger data is loaded correctly from an AIAA benchmark file.

    Test for deprecation warning when using from_file attribute

    Parameters
    ----------
    aiaa_bechmark_trigger_file : str
        Path to the AIAA benchmark file with trigger data (fixture).
    """
    with pytest.deprecated_call():
        trigger_data = ac.tools.TriggerAIAABenchmark(name=aiaa_bechmark_trigger_file)
    trigger_data = ac.tools.TriggerAIAABenchmark(file=aiaa_bechmark_trigger_file)
    assert trigger_data.numchannels == 1
    assert trigger_data.numsamples == 480
    assert trigger_data.sample_freq == 48000
    assert trigger_data.data.shape == (trigger_data.numsamples, trigger_data.numchannels)
