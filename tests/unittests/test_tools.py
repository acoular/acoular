import numpy as np
import pytest
from acoular.tools import c_air


def test_c_air_value():
    # Test if the function produces a certain value
    celsius = 20
    h = 0.5
    p = 101325
    c_co2 = 0.04
    expected_speed_of_sound = 344.01680167676795
    calculated_speed_of_sound = c_air(celsius, h, p, c_co2)
    assert np.isclose(calculated_speed_of_sound, expected_speed_of_sound)


def test_c_air_value_error():
    # Test if the function raises ValueError for out of range temperature
    with pytest.raises(ValueError, match='Temperature out of range'):
        c_air(-10, 0.5, 101325, 0.04)
    with pytest.raises(ValueError, match='Temperature out of range'):
        c_air(40, 0.5, 101325, 0.04)

    # Test if the function raises ValueError for out of range pressure
    with pytest.raises(ValueError, match='Pressure out of range'):
        c_air(20, 0.5, 50000, 0.04)
    with pytest.raises(ValueError, match='Pressure out of range'):
        c_air(20, 0.5, 120000, 0.04)

    # Test if the function raises ValueError for out of range water vapor mole fraction
    with pytest.raises(ValueError, match='Water vapor mole fraction out of range'):
        c_air(20, 1.1, 101325, 0.04)

    # Test if the function raises ValueError for out of range CO2 concentration
    with pytest.raises(ValueError, match='CO2 concentration out of range'):
        c_air(20, 0.5, 101325, 1.5)
