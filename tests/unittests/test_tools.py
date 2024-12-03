import re

import acoular as ac
import numpy as np
import pytest
from acoular.tools import c_air
from pytest_cases import parametrize


def test_c_air_value():
    t = 20
    h = 10
    p = 101325
    c_co2 = 0.04
    expected_speed_of_sound = 343.49100070169663
    calculated_speed_of_sound = c_air(t, h, p, c_co2)
    assert np.isclose(calculated_speed_of_sound, expected_speed_of_sound)


@parametrize(
    't, h, p, co2',
    [
        (-1, 10, 101325, 0.04),
        (31, 10, 101325, 0.04),
        (20, 10, 59999, 0.04),
        (20, 10, 110001, 0.04),
        (20, 10, 101325, 1.01),
    ],
    ids=[
        'Temperature less than 0',
        'Temperature greater than 30',
        'Pressure less than 60 kPa',
        'Pressure greater than 110 kPa',
        'CO2 concentration greater than 1%',
    ],
)
def test_value_error(t, h, p, co2):
    """Test if function raises errors for the given input values."""
    if t < 0 or t > 30:
        match = 'Temperature out of range (0°C to 30°C)'
    elif p < 60000 or p > 110000:
        match = 'Pressure out of range (60 kPa to 110 kPa)'
    elif co2 > 1.0:
        match = 'CO2 concentration out of range (up to 1%)'

    with pytest.raises(ValueError, match=re.escape(match)):
        c_air(t=t, h=h, p=p, co2=co2)


@pytest.mark.xfail(strict=True)
def test_cart_cyl_transform():
    """Tests if transformation between cartesian and cylindrical coordinates are consistent."""
    original_cartesian = np.abs(np.random.RandomState(1).rand(3))
    converted_cartesian = ac.cylToCart(ac.cartToCyl(original_cartesian))
    np.testing.assert_allclose(converted_cartesian, original_cartesian)
