from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize


@parametrize('acoular_cls, suffix', [(ac.WriteWAV, '.wav'), (ac.WriteH5, '.h5')], ids=['WriteWAV', 'WriteH5'])
def test_deprecation_warnings(tmp_path, acoular_cls, suffix):
    """Test that deprecation warnings are raised when using the 'name' argument.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory to store the test file (pytest fixture).
    acoular_cls : cls
        Acoular class to test.
    suffix : str
        File suffix to use for the test file.
    """
    file_path = (tmp_path / 'test_file').with_suffix(suffix)
    with pytest.deprecated_call():
        acoular_cls(name=file_path)
    acoular_cls(file=file_path)


def test_td_dir(tmp_path, create_time_data_source):
    """Test that global time data directory is used when no filename is given."""
    ac.config.td_dir = tmp_path
    time_data = create_time_data_source(num_channels=2, num_samples=1)
    h5 = ac.WriteH5(source=time_data)
    h5.save()
    assert (tmp_path / Path(h5.file).name).exists()


@parametrize('h5library', ['pytables', 'h5py'])
@parametrize('data', ['test', 10, np.random.randn(3)], ids=['str', 'scalar', 'array'])
@parametrize('acoular_cls', [ac.TimeSamples, ac.MaskedTimeSamples])
def test_metadata(tmp_path, create_time_data_source, acoular_cls, h5library, data):
    """Test that metadata can be saved in the h5 file and loaded correctly."""
    ac.config.h5library = h5library
    # save metadata in h5 file
    ac.config.td_dir = tmp_path
    time_data = create_time_data_source(num_channels=2, num_samples=1)
    h5 = ac.WriteH5(source=time_data, metadata={'test': data})
    h5.save()
    # load metadata from h5 file
    ts = acoular_cls(file=h5.file)
    if isinstance(data, (float, int)):
        assert ts.metadata['test'][()] == data
    if isinstance(data, str):
        assert ts.metadata['test'][()] == data.encode()
    else:
        np.testing.assert_allclose(ts.metadata['test'][()], data)
