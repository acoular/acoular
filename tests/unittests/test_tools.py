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


# Polygon tests
class TestPolygon:
    """Test cases for the Polygon class in tools.utils."""

    def test_polygon_mismatched_lengths(self):
        """Test that Polygon raises IndexError when x and y have different lengths."""
        from acoular.tools.utils import Polygon

        with pytest.raises(IndexError, match='x and y must be equally sized'):
            Polygon([0, 1, 2], [0, 1])

    def test_polygon_scalar_input(self):
        """Test Polygon.is_inside with scalar inputs."""
        from acoular.tools.utils import Polygon

        # Create a square polygon
        poly = Polygon([0, 1, 1, 0], [0, 0, 1, 1])
        
        # Test point inside (scalar)
        result = poly.is_inside(0.5, 0.5)
        assert isinstance(result, float)
        assert result > 0
        
        # Test point outside (scalar)
        result = poly.is_inside(2.0, 2.0)
        assert isinstance(result, float)
        assert result < 0
        
        # Test point on edge (scalar)
        result = poly.is_inside(0.5, 0.0)
        assert isinstance(result, float)
        assert abs(result) < 1e-10

    def test_polygon_mismatched_point_shapes(self):
        """Test that is_inside raises IndexError when xpoint and ypoint have different shapes."""
        from acoular.tools.utils import Polygon

        poly = Polygon([0, 1, 1, 0], [0, 0, 1, 1])
        
        with pytest.raises(IndexError, match='x and y has different shapes'):
            poly.is_inside(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5]))

    def test_polygon_clockwise_to_counterclockwise(self):
        """Test that clockwise coordinates are converted to counter-clockwise."""
        from acoular.tools.utils import Polygon

        # Define polygon in clockwise order
        x_cw = [0, 0, 1, 1]
        y_cw = [0, 1, 1, 0]
        poly_cw = Polygon(x_cw, y_cw)
        
        # Define polygon in counter-clockwise order
        x_ccw = [0, 1, 1, 0]
        y_ccw = [0, 0, 1, 1]
        poly_ccw = Polygon(x_ccw, y_ccw)
        
        # Test that both give same result for a point inside
        test_x, test_y = 0.5, 0.5
        result_cw = poly_cw.is_inside(test_x, test_y)
        result_ccw = poly_ccw.is_inside(test_x, test_y)
        
        assert result_cw > 0
        assert result_ccw > 0

    def test_polygon_open_gets_closed(self):
        """Test that open polygons are automatically closed."""
        from acoular.tools.utils import Polygon

        # Define open polygon (last point != first point)
        x_open = [0, 1, 1, 0]
        y_open = [0, 0, 1, 1]
        poly = Polygon(x_open, y_open)
        
        # Check that polygon was closed (first and last points are the same)
        assert poly.x[0] == poly.x[-1]
        assert poly.y[0] == poly.y[-1]

    def test_polygon_already_closed(self):
        """Test polygon that is already closed."""
        from acoular.tools.utils import Polygon

        # Define closed polygon (last point == first point)
        x_closed = [0, 1, 1, 0, 0]
        y_closed = [0, 0, 1, 1, 0]
        poly = Polygon(x_closed, y_closed)
        
        # Should remain the same length
        assert len(poly.x) == 5
        assert len(poly.y) == 5

    def test_polygon_array_input(self):
        """Test Polygon.is_inside with array inputs."""
        from acoular.tools.utils import Polygon

        poly = Polygon([0, 1, 1, 0], [0, 0, 1, 1])
        
        # Test multiple points
        xpoints = np.array([0.5, 2.0, 0.5])
        ypoints = np.array([0.5, 2.0, 0.0])
        results = poly.is_inside(xpoints, ypoints)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert results[0] > 0  # inside
        assert results[1] < 0  # outside
        assert abs(results[2]) < 1e-10  # on edge

    def test_polygon_custom_tolerance(self):
        """Test Polygon.is_inside with custom tolerance."""
        from acoular.tools.utils import Polygon

        poly = Polygon([0, 1, 1, 0], [0, 0, 1, 1])
        
        # Point on edge with larger tolerance
        result = poly.is_inside(0.5, 1e-11, smalld=1e-10)
        assert abs(result) < 1e-9

    def test_polygon_complex_shape(self):
        """Test Polygon with a more complex shape."""
        from acoular.tools.utils import Polygon

        # Create a pentagon
        angles = np.linspace(0, 2*np.pi, 6)
        x_pentagon = np.cos(angles)
        y_pentagon = np.sin(angles)
        poly = Polygon(x_pentagon, y_pentagon)
        
        # Test center point (should be inside)
        result = poly.is_inside(0.0, 0.0)
        assert result > 0
        
        # Test far point (should be outside)
        result = poly.is_inside(5.0, 5.0)
        assert result < 0

    def test_polygon_concave_polygon(self):
        """Test Polygon with a concave shape (L-shape)."""
        from acoular.tools.utils import Polygon

        # L-shaped polygon
        x_l = [0, 2, 2, 1, 1, 0]
        y_l = [0, 0, 1, 1, 2, 2]
        poly = Polygon(x_l, y_l)
        
        # Point inside the L
        result = poly.is_inside(0.5, 0.5)
        assert result > 0
        
        # Point in the "notch" (should be outside)
        result = poly.is_inside(1.5, 1.5)
        assert result < 0
