import acoular as ac


def test_load_mics(aiaa_bechmark_time_data_file):
    """Test if microphone positions are loaded correctly from an AIAA benchmark file.

    Parameters
    ----------
    aiaa_bechmark_time_data_file : str
        Path to the AIAA benchmark file with time data (fixture).

    """
    mic_geom = ac.aiaa.MicAIAABenchmark(file=aiaa_bechmark_time_data_file)
    assert mic_geom.num_mics == 8
    assert mic_geom.pos.shape == (3, 8)
    assert mic_geom.pos_total.shape == (3, 8)


def test_load_csm(aiaa_bechmark_csm_file):
    """Test if cross spectral matrix is loaded correctly from an AIAA benchmark file.

    Parameters
    ----------
    aiaa_bechmark_csm_file : str
        Path to the AIAA benchmark file with cross spectral matrix data (fixture
    """
    csm = ac.aiaa.CsmAIAABenchmark(file=aiaa_bechmark_csm_file)
    assert csm.num_channels == 8
    assert csm.csm.shape == (csm.fftfreq().shape[0], csm.num_channels, csm.num_channels)


def test_load_time(aiaa_bechmark_time_data_file):
    """Test if time data is loaded correctly from an AIAA benchmark file.

    Parameters
    ----------
    aiaa_bechmark_time_data_file : str
        Path to the AIAA benchmark file with time data (fixture).
    """
    time_data = ac.aiaa.TimeSamplesAIAABenchmark(file=aiaa_bechmark_time_data_file)
    assert time_data.num_channels == 8
    assert time_data.num_samples == 480
    assert time_data.sample_freq == 48000
    assert time_data.data.shape == (time_data.num_samples, time_data.num_channels)
    assert time_data.metadata != {}


def test_load_trigger(aiaa_bechmark_trigger_file):
    """Test if trigger data is loaded correctly from an AIAA benchmark file.

    Parameters
    ----------
    aiaa_bechmark_trigger_file : str
        Path to the AIAA benchmark file with trigger data (fixture).
    """
    trigger_data = ac.aiaa.TriggerAIAABenchmark(file=aiaa_bechmark_trigger_file)
    assert trigger_data.num_channels == 1
    assert trigger_data.num_samples == 480
    assert trigger_data.sample_freq == 48000
    assert trigger_data.data.shape == (trigger_data.num_samples, trigger_data.num_channels)
