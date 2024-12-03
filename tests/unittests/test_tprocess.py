from pathlib import Path

import acoular as ac
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


def test_td_dir(tmp_path, time_data_source):
    """Test that global time data directory is used when no filename is given."""
    ac.config.td_dir = tmp_path
    h5 = ac.WriteH5(source=time_data_source)
    h5.save()
    assert (tmp_path / Path(h5.file).name).exists()
