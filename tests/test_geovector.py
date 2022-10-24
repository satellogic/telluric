import math
import tempfile

import pytest
import shapely
from packaging import version
from unittest import mock
from numpy.testing import assert_array_almost_equal
from pytest import approx

from shapely.geometry import Point, Polygon, mapping, LineString, CAP_STYLE

from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform_bounds

from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_UNARY_OPERATIONS, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS,
    generate_tile_coordinates, get_dimension,
    generate_tile_coordinates_from_pixels)
from telluric.constants import DEFAULT_CRS, WGS84_CRS, WEB_MERCATOR_CRS


def test_geovector_has_shape_and_default_crs():
    shape = Point(0.0, 0.0)
    gv = GeoVector(shape)

    assert gv.get_shape(gv.crs) == shape
    assert gv.crs == DEFAULT_CRS


def test_geovector_has_given_crs():
    crs = CRS({'init': 'epsg:4326'})
    gv = GeoVector(None, crs)

    assert gv.crs == crs


def test_geovector_representation():
    shape = Point(0.0, 0.0)
    gv = GeoVector(shape)

    assert str(gv) == "GeoVector(shape=POINT (0 0), crs=EPSG:4326)"


def test_geovector_from_bounds_has_proper_shape():
    shape = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])

    gv1 = GeoVector.from_bounds(xmin=0, ymin=0, xmax=1, ymax=1)
    gv2 = GeoVector.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
    gv3 = GeoVector.from_bounds(xmax=1, ymax=1, xmin=0, ymin=0)

    assert gv1 == gv2 == gv3
    assert gv1.get_shape(gv1.crs) == shape


def test_geovector_has_proper_bounds():
    shape = Point(0.0, 0.0).buffer(1.0)
    gv = GeoVector(shape)

    bounds = gv.get_bounds(gv.crs)

    assert bounds.left == gv.left == -1
    assert bounds.bottom == gv.bottom == -1
    assert bounds.right == gv.right == 1
    assert bounds.top == gv.top == 1

    xmin, ymin, xmax, ymax = bounds

    assert (xmin, ymin, xmax, ymax) == (-1, -1, 1, 1)


def test_geovector_from_geojson():
    gv = GeoVector.from_geojson("tests/data/vector/simple_vector.json")

    assert gv.crs == WGS84_CRS
    assert gv.get_shape(gv.crs).type == 'Polygon'


def test_geovector_to_from_geojson():
    gv = GeoVector.from_bounds(xmin=0, ymin=0, xmax=1, ymax=1)

    with tempfile.NamedTemporaryFile('w') as fp:
        gv.to_geojson(fp.name)

        assert GeoVector.from_geojson(fp.name) == gv


def test_geovector_record_transformation():
    gv = GeoVector.from_geojson("tests/data/vector/simple_vector.json")

    record = gv.to_record(WGS84_CRS)

    gv_deserialized = GeoVector.from_record(record, WGS84_CRS)

    assert gv == gv_deserialized


def test_geovector_from_xyz():
    gv = GeoVector.from_xyz(0, 0, 0)

    assert gv.crs == WEB_MERCATOR_CRS

    xmin, ymin, xmax, ymax = gv.get_bounds(WGS84_CRS)

    assert xmin == approx(-180)
    assert xmax == approx(180)
    assert ymin == approx(-85.051129)
    assert ymax == approx(85.051129)


def test_get_bounding_box():
    src_crs = CRS(init='epsg:4326')
    dst_crs = CRS(init='epsg:32718')
    src_bounds = dict(xmin=-73.309037, ymin=-40.665865,
                      xmax=-72.723835, ymax=-40.026434)

    gv = GeoVector.from_bounds(crs=src_crs, **src_bounds)
    bounds = transform_bounds(src_crs=src_crs, dst_crs=dst_crs,
                              left=src_bounds['xmin'], bottom=src_bounds['ymin'],
                              right=src_bounds['xmax'], top=src_bounds['ymax'])

    assert gv.get_bounding_box(dst_crs).almost_equals(
        GeoVector.from_bounds(*bounds, crs=dst_crs))


def test_reproject_changes_crs():
    shape = Point(0.0, 40.0)
    new_crs = CRS({'init': 'epsg:32630'})

    gv = GeoVector(shape)

    new_gv = gv.reproject(new_crs)

    assert new_gv.crs == new_crs


def test_reproject_same_projection_returns_same_object():
    shape = Point(0.0, 0.0)

    gv = GeoVector(shape)

    new_gv = gv.reproject(gv.crs)

    assert new_gv is gv


def test_reproject_respects_units():
    # See https://publicgitlab.satellogic.com/telluric/telluric/issues/87

    # https://epsg.io/32038
    crs = {
        'proj': 'lcc',
        'lat_0': 31.66666666666667,
        'units': 'us-ft',
        'x_0': 609601.2192024384,
        'lat_2': 33.96666666666667,
        'datum': 'NAD27',
        'no_defs': True,
        'y_0': 0,
        'lon_0': -97.5,
        'lat_1': 32.13333333333333
    }
    gv = GeoVector.from_bounds(
        xmin=-1000, xmax=10000, ymin=-10000, ymax=10000, crs=crs
    )

    # Obtained from geojson.io
    expected_gv = GeoVector(
        Polygon([
            [-103.9206783868097, 31.47123320311564],
            [-103.9245979236265, 31.52609724089042],
            [-103.8893692309087, 31.52793986790464],
            [-103.8854710802219, 31.47307467582068],
            [-103.9206783868097, 31.47123320311564]
        ]),
        WGS84_CRS
    )

    # Equality fails on the last decimal places depending on the platform
    # assert gv.reproject(WGS84_CRS) == expected_gv
    assert gv.reproject(WGS84_CRS).equals_exact(expected_gv, 1e-12)


def test_area_is_cartesian_and_correct():
    shape = Point(0.0, 40.0).buffer(1.0)
    expected_area = 29773634861.23  # geojson.io

    gv = GeoVector(shape)

    assert gv.area == approx(expected_area)


def test_svg_contains_geometry():
    shape = Point(0.0, 10.0)

    expected_substring = '<circle cx="0.0" cy="10.0"'

    gv = GeoVector(shape)

    assert expected_substring in gv._repr_svg_()


def test_geo_interface():
    shape = Point(0.0, 10.0)
    crs = {'init': 'epsg:32630'}

    expected_geo_interface = {'coordinates': (-7.488743884392727, 0.0), 'type': 'Point'}

    gv = GeoVector(shape, crs)

    geo_interface = mapping(gv)

    assert geo_interface.keys() == expected_geo_interface.keys()
    assert geo_interface['type'] == expected_geo_interface['type']
    assert geo_interface['coordinates'][0] == approx(expected_geo_interface['coordinates'][0])
    assert geo_interface['coordinates'][1] == approx(expected_geo_interface['coordinates'][1], abs=1e-4)


def test_almost_equals():
    some_crs = CRS({'init': 'epsg:32630'})
    another_crs = CRS({'init': 'epsg:32631'})

    pt = GeoVector(Point(0, 0), some_crs)

    similar = GeoVector(Point(0, 0.00000000001), some_crs)
    assert pt.almost_equals(similar.reproject(another_crs))

    different = GeoVector(Point(0, 1), some_crs)
    assert pt != different


def test_equality_different_topology():
    gv1 = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    gv2 = GeoVector(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

    assert gv1 == gv2


def test_print():
    vector = GeoVector(Point(0.0, 0.0))
    assert '%s' % vector.crs in '%s' % vector
    assert '%s' % vector.get_shape(vector.crs) in '%s' % vector


def test_centroid():
    vector = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    expected_centroid = GeoVector(Point(0.5, 0.5))

    assert vector.centroid == expected_centroid


@pytest.mark.parametrize("property_name", GEOM_PROPERTIES)
def test_delegated_properties(property_name):
    vector = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    assert getattr(vector, property_name).get_shape(vector.crs) == getattr(vector.get_shape(vector.crs), property_name)


@pytest.mark.parametrize("property_name", ['x', 'y', 'xy'])
def test_delegated_properties(property_name):
    vector = GeoVector(Point(0, 10))

    assert getattr(vector, property_name) == getattr(vector.get_shape(vector.crs), property_name)


@pytest.mark.parametrize("predicate_name", GEOM_UNARY_PREDICATES)
def test_delegated_unary_predicates(predicate_name):
    vector = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    assert getattr(vector, predicate_name) == getattr(vector.get_shape(vector.crs), predicate_name)


@pytest.mark.parametrize("operation_name", GEOM_UNARY_OPERATIONS)
@pytest.mark.parametrize("test_param", [0.1, 0.5, 1, 2, 10])
def test_delegated_unary_predicates(operation_name, test_param):
    vector = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    assert (getattr(vector, operation_name)(test_param) ==
            GeoVector(getattr(vector.get_shape(vector.crs), operation_name)(test_param), vector.crs))


@pytest.mark.parametrize("predicate_name", GEOM_BINARY_PREDICATES)
def test_delegated_binary_predicates(predicate_name):
    vector_1 = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    vector_2 = GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]))

    assert (getattr(vector_1, predicate_name)(vector_2) ==
            getattr(vector_1.get_shape(vector_1.crs), predicate_name)(vector_2.get_shape(vector_2.crs)))


@pytest.mark.parametrize("predicate_name", ["intersects"])
def test_delegated_binary_predicates_for_impossible_transformations(predicate_name):
    vector_1 = GeoVector.from_bounds(-180, -90, 180, 90, crs=CRS(init='epsg:4326'))
    vector_2 = GeoVector.from_bounds(-1000, -1000, 1000, 1000, crs=CRS(init='epsg:3857'))

    assert getattr(vector_2, predicate_name)(vector_1)


@pytest.mark.parametrize("operation_name", GEOM_BINARY_OPERATIONS)
def test_delegated_operations(operation_name):
    vector_1 = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    vector_2 = GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]))

    result_vector = getattr(vector_1, operation_name)(vector_2)

    assert (result_vector.get_shape(result_vector.crs) ==
            getattr(vector_1.get_shape(vector_1.crs), operation_name)(vector_2.get_shape(vector_2.crs)))


def test_generate_tile_coordinates():
    roi = GeoVector(Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]))
    num_tiles = (10, 10)

    tiles = list(generate_tile_coordinates(roi, num_tiles))

    assert len(tiles) == 100

    for ii, tile in enumerate(tiles):
        shape = tile.get_shape(tile.crs)
        assert_array_almost_equal(shape.bounds[:2], (0.0 + (ii % 10) * 0.1, 0.0 + (ii // 10) * 0.2))
        assert_array_almost_equal(shape.bounds[2:], (0.1 + (ii % 10) * 0.1, 0.2 + (ii // 10) * 0.2))

        assert tile.crs == roi.crs


@pytest.mark.parametrize("pixel_size,resolution,length", [
    ((1, 1), 1, 200),  # 10 x 20
    ((1, 1), 2, 50),  # 5 x 10
    ((1, 1), 0.5, 800),  # 20 x 40
    ((1, 1), 1.5, 98),  # (6 + 1) x (13 + 1)
    ((9, 19), 1, 4),  # (1 + 1) x (1 + 1)
    ((9, 19), 2, 1),  # 1 x 1
    ((11, 21), 1, 1),  # 1 x 1
])
def test_generate_tile_coordinates_in_pixels(pixel_size, resolution, length):
    roi = GeoVector(Polygon([(0, 0), (10, 0), (10, 20), (0, 20)]))

    tiles = list(generate_tile_coordinates_from_pixels(roi, resolution, pixel_size))

    assert len(tiles) == length


def test_generate_tile_coordinates_in_pixels_raises_error_for_non_int_pixel_size():
    roi = GeoVector(Polygon([(0, 0), (10, 0), (10, 20), (0, 20)]))

    with pytest.raises(ValueError) as error:
        list(generate_tile_coordinates_from_pixels(roi, 1, (1.5, 1.5)))

    assert "Pixel size must be a tuple of integers" in error.exconly()


def test_get_dimension():
    # Source: https://en.wikipedia.org/wiki/GeoJSON#Geometries
    data = {
        "type": "Point",
        "coordinates": [30, 10]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "Point",
        "coordinates": [30, 10, 1]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "LineString",
        "coordinates": [
            [30, 10], [10, 30], [40, 40]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "LineString",
        "coordinates": [
            [30, 10, 1], [10, 30, 1], [40, 40, 1]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "Polygon",
        "coordinates": [
            [[30, 10], [40, 40], [20, 40], [10, 20], [30, 10]]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "Polygon",
        "coordinates": [
            [[30, 10, 1], [40, 40, 1], [20, 40, 1], [10, 20, 1], [30, 10, 1]]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "Polygon",
        "coordinates": [
            [[35, 10], [45, 45], [15, 40], [10, 20], [35, 10]],
            [[20, 30], [35, 35], [30, 20], [20, 30]]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "Polygon",
        "coordinates": [
            [[35, 10, 1], [45, 45, 1], [15, 40, 1], [10, 20, 1], [35, 10, 1]],
            [[20, 30, 1], [35, 35, 1], [30, 20, 1], [20, 30, 1]]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "MultiPoint",
        "coordinates": [
            [10, 40], [40, 30], [20, 20], [30, 10]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "MultiPoint",
        "coordinates": [
            [10, 40, 1], [40, 30, 1], [20, 20, 1], [30, 10, 1]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "MultiLineString",
        "coordinates": [
            [[10, 10], [20, 20], [10, 40]],
            [[40, 40], [30, 30], [40, 20], [30, 10]]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "MultiLineString",
        "coordinates": [
            [[10, 10, 1], [20, 20, 1], [10, 40, 1]],
            [[40, 40, 1], [30, 30, 1], [40, 20, 1], [30, 10, 1]]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[30, 20], [45, 40], [10, 40], [30, 20]]
            ],
            [
                [[15, 5], [40, 10], [10, 20], [5, 10], [15, 5]]
            ]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[30, 20, 1], [45, 40, 1], [10, 40, 1], [30, 20, 1]]
            ],
            [
                [[15, 5, 1], [40, 10, 1], [10, 20, 1], [5, 10, 1], [15, 5, 1]]
            ]
        ]
    }
    assert get_dimension(data) == 3

    data = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[40, 40], [20, 45], [45, 30], [40, 40]]
            ],
            [
                [[20, 35], [10, 30], [10, 10], [30, 5], [45, 20], [20, 35]],
                [[30, 20], [20, 15], [20, 25], [30, 20]]
            ]
        ]
    }
    assert get_dimension(data) == 2

    data = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[40, 40, 1], [20, 45, 1], [45, 30, 1], [40, 40, 1]]
            ],
            [
                [[20, 35, 1], [10, 30, 1], [10, 10, 1], [30, 5, 1], [45, 20, 1], [20, 35, 1]],
                [[30, 20, 1], [20, 15, 1], [20, 25, 1], [30, 20, 1]]
            ]
        ]
    }
    assert get_dimension(data) == 3


def test_polygonize_line():
    diag = 1 / math.sqrt(2)

    line = GeoVector(LineString([(0, 0), (1, 1)]))
    expected_result = GeoVector(Polygon([
        (1 - diag / 2, 1 + diag / 2), (1 + diag / 2, 1 - diag / 2), (diag / 2, -diag / 2), (-diag / 2, diag / 2)
    ]))

    result = line.polygonize(1)

    assert result == expected_result


@pytest.mark.skipif(
    version.parse(shapely.__version__) >= version.parse('1.7b1'),
    reason="Shapely < 1.7b1 is required, https://github.com/satellogic/telluric/issues/269",
)
def test_polygonize_line_square_cap_style():
    diag = 1 / math.sqrt(2)

    line = GeoVector(LineString([(0, 0), (1, 1)]))
    expected_result = GeoVector(Polygon([
        (1, 1 + diag), (1 + diag, 1), (0, -diag), (-diag, 0)
    ]))

    result = line.polygonize(1, cap_style_line=CAP_STYLE.square)

    # Don't use result == expected_result as topology might be different
    assert result.equals(expected_result)


def test_polygonize_point():
    point = GeoVector(Point([0, 0]))
    expected_bounds = (-0.5, -0.5, 0.5, 0.5)
    expected_area = math.pi * 0.5 ** 2

    result = point.polygonize(1)

    result_shape = result.get_shape(result.crs)

    assert result_shape.bounds == expected_bounds
    assert result_shape.area == approx(expected_area, rel=1e-2)


def test_tiles():
    aoi = GeoVector(
        Polygon([
            [109.612458654485849, 40.867336965196827],
            [109.619116438865731, 40.867336965196827],
            [109.61872894876953, 40.864459877761597],
            [109.612599559975365, 40.864326676241866],
            [109.612458654485849, 40.867336965196827]
        ]),
        WGS84_CRS
    )
    tiles = list(aoi.tiles(15))
    assert tiles == [(26361, 12301, 15)]
    # multiple resolutions, multiple tiles
    tiles = list(aoi.tiles([15, 16]))
    assert tiles == [(26361, 12301, 15), (52722, 24603, 16), (52723, 24603, 16)]


@mock.patch('telluric.rasterization.rasterize')
def test_rasterize_without_bounds(mock_rasterize):
    gv = GeoVector(Polygon.from_bounds(0, 0, 1, 1))
    gv.rasterize(dest_resolution=0.1, fill_value=29)
    expected_shape = [gv.get_shape(gv.crs)]
    expected_bounds = gv.envelope.get_shape(gv.crs)
    mock_rasterize.assert_called_with(expected_shape, gv.crs,
                                      expected_bounds, 0.1,
                                      fill_value=29, dtype=None)


@mock.patch('telluric.rasterization.rasterize')
def test_rasterize_with_geovector_bounds(mock_rasterize):
    gv = GeoVector(Polygon.from_bounds(0, 0, 1, 1))
    expected_bounds = Polygon.from_bounds(0.1, 0.1, 2, 2)
    bounds = GeoVector(expected_bounds, crs=gv.crs)
    gv.rasterize(0.00001, bounds=bounds)
    expected_shape = [gv.get_shape(gv.crs)]
    mock_rasterize.assert_called_with(expected_shape, gv.crs,
                                      expected_bounds, 0.00001,
                                      fill_value=None, dtype=None)


@mock.patch('telluric.rasterization.rasterize')
def test_rasterize_with_polygon_bounds(mock_rasterize):
    gv = GeoVector(Polygon.from_bounds(0, 0, 1, 1))
    expected_bounds = Polygon.from_bounds(0.1, 0.1, 2, 2)
    bounds = expected_bounds
    gv.rasterize(0.00001, bounds=bounds)
    expected_shape = [gv.get_shape(gv.crs)]
    mock_rasterize.assert_called_with(expected_shape, gv.crs,
                                      expected_bounds, 0.00001,
                                      fill_value=None, dtype=None)


@mock.patch('telluric.rasterization.rasterize')
def test_rasterize_with_crs(mock_rasterize):
    gv = GeoVector(Polygon.from_bounds(0, 0, 1, 1))
    expected_crs = WEB_MERCATOR_CRS
    expected_bounds = Polygon.from_bounds(11132, 11132, 222639, 222684)
    bounds = GeoVector(Polygon.from_bounds(0.1, 0.1, 2, 2))
    gv.rasterize(1000, bounds=bounds, crs=WEB_MERCATOR_CRS)
    expected_shape = [gv.get_shape(expected_crs)]

    args, kwargs = mock_rasterize.call_args
    assert args[0] == expected_shape
    assert args[1] == expected_crs
    assert args[2].equals_exact(expected_bounds, tolerance=0.5)


def test_geovector_empty_is_empty():
    gv = GeoVector.empty()
    assert gv.is_empty


def test_geovector_quick_operations():
    gv1 = GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=1)
    gv2 = GeoVector.from_bounds(xmin=1, ymin=0, xmax=3, ymax=1)

    assert (gv1 | gv2) == GeoVector.from_bounds(xmin=0, ymin=0, xmax=3, ymax=1)
    assert (gv1 & gv2) == GeoVector.from_bounds(xmin=1, ymin=0, xmax=2, ymax=1)


def test_geovector_copy():
    gv1 = GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=1)
    gv1_copy = gv1.copy()
    assert gv1 == gv1_copy
    assert id(gv1) != id(gv1_copy)
