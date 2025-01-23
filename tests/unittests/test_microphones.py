from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from pytest_cases import fixture, parametrize


@fixture(scope='module')
@parametrize('invalid_channels', [[], [0, 2]], ids=['no_invalid_channels', 'invalid_channels'])
def micgeom(invalid_channels):
    """Fixture for creating a rectangular MicGeom object with 4 microphones."""
    # create 4 points with center at (0, 0, 0) and aperture of 2
    pos_total = np.array([[0, 1, 0, -1], [1, 0, -1, 0], [0, 0, 0, 0]])
    return ac.MicGeom(pos_total=pos_total, invalid_channels=invalid_channels)


def test_load_xml():
    """Test if calibration data is loaded correctly from an XML file.

    Test for deprecation warning when using from_file attribute
    """
    xml_file_path = Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml'
    # Test for deprecation warning when using from_file attribute
    with pytest.deprecated_call():
        mic_geom = ac.MicGeom(from_file=xml_file_path)
    # Test for correct number of microphones and shapes
    mic_geom = ac.MicGeom(file=xml_file_path)
    assert mic_geom.num_mics == 64
    assert mic_geom.pos.shape == (3, 64)
    assert mic_geom.pos_total.shape == (3, 64)


def test_export_mpos(tmp_path, micgeom):
    """Test if the microphone positions are exported correctly to an XML file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory to store the exported file (fixture).
    micgeom : acoular.MicGeom
        MicGeom object with 4 microphones (fixture).
    """
    # Export the microphone positions to a temporary file
    export_file = tmp_path / 'test_mpos.xml'
    micgeom.export_mpos(export_file)
    # Load and check if the digests match
    new_mic_geom = ac.MicGeom(file=export_file)
    assert micgeom.digest == new_mic_geom.digest


def test_mpos_center(micgeom):
    """Test if the center of the microphone positions is calculated correctly.

    Parameters
    ----------
    micgeom : acoular.MicGeom
        MicGeom object with 4 microphones (fixture).
    """
    expected_center = np.array([0, 0, 0])
    assert np.allclose(micgeom.center, expected_center), f'Expected {expected_center}, but got {micgeom.center}'


def test_mpos_aperture(micgeom):
    """Test if the aperture of the microphone positions is calculated correctly.

    Parameters
    ----------
    micgeom : acoular.MicGeom
        MicGeom object with 4 microphones (fixture).
    """
    expected_aperture = 2.0
    assert np.allclose(
        micgeom.aperture, expected_aperture
    ), f'Expected aperture {expected_aperture}, but got {micgeom.aperture}'
