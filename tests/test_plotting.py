import pytest
from unittest import mock

from shapely.geometry import shape, Polygon, Point
from folium import GeoJson

from telluric.vectors import GeoVector
from telluric.plotting import layer_from_element, plot


def test_layer_from_shape():
    expected_geo = {'coordinates': (0.0, 0.0), 'type': 'Point'}
    vector = GeoVector(shape(expected_geo))

    result = layer_from_element(vector)

    assert result.data == expected_geo


def test_plot_empty_geometry_prints_warning():
    vector = GeoVector(Point([0, 0]).buffer(0))
    assert vector.is_empty

    with pytest.warns(UserWarning) as record:
        plot(vector)

    assert "The geometry is empty." in record[0].message.args[0]


@mock.patch.object(GeoJson, "add_to")
def test_plot(mock_add_to):
    gv = GeoVector(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

    plot(gv)

    mock_add_to.assert_called_once()
