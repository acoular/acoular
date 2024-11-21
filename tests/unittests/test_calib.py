from pathlib import Path

import acoular as ac
import pytest


def test_load_xml():
    """Test if calibration data is loaded correctly from an XML file.

    Test for deprecation warning when using from_file attribute
    """
    xml_file_path = Path(__file__).parent.parent.parent / 'examples' / 'data' / 'example_calib.xml'
    with pytest.deprecated_call():
        calib = ac.Calib(from_file=xml_file_path)
    calib = ac.Calib(file=xml_file_path)
    assert calib.num_mics == 56
    assert calib.data.shape == (56,)
