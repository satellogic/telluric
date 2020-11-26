import pytest
from unittest import mock

pytest.importorskip("folium")  # noqa: E402

from shapely.geometry import shape, Polygon, Point
from ipyleaflet import Map

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


@mock.patch.object(Map, "add_layer")
@mock.patch("telluric.plotting.layer_from_element", autospec=True)
def test_plot_adds_layer_to_map(mock_layer_from_element, mock_add_layer):
    gv = GeoVector(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    layer = mock_layer_from_element.return_value

    plot(gv)

    mock_add_layer.assert_called_once_with(layer)
