import pytest

import numpy as np
from affine import Affine
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon, LineString, Point

from rasterio.crs import CRS

from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.georaster import GeoRaster2
from telluric.collections import FeatureCollection
from telluric.constants import DEFAULT_CRS, WEB_MERCATOR_CRS, WGS84_CRS

from telluric.rasterization import ScaleError, rasterize


def test_rasterization_raise_error_for_too_small_image():
    shape = Polygon([(0, 0), (1, 0), (1, -1), (0, -1)])
    fcol = FeatureCollection([GeoFeature(GeoVector(shape), {})])

    with pytest.raises(ScaleError) as excinfo:
        fcol.rasterize(1e10)
    assert "Scale is too coarse, decrease it for a bigger image" in excinfo.exconly()


def test_rasterization_raise_error_for_too_big_image():
    shape = Polygon([(0, 0), (1, 0), (1, -1), (0, -1)])
    fcol = FeatureCollection([GeoFeature(GeoVector(shape), {})])

    with pytest.raises(ScaleError) as excinfo:
        fcol.rasterize(1e-50)
    assert "Scale is too fine, increase it for a smaller image" in excinfo.exconly()


def test_rasterization_has_expected_affine_and_crs():
    shape = Polygon([(0, 0), (1, 0), (1, -1), (0, -1)])
    crs = CRS({'init': 'epsg:32631'})
    fcol = FeatureCollection([GeoFeature(GeoVector(shape, crs), {})])

    expected_affine = ~Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    georaster = fcol.rasterize(1, crs=crs)

    assert georaster.crs == crs

    assert georaster.affine.a == pytest.approx(expected_affine.a)
    assert georaster.affine.b == pytest.approx(expected_affine.b, abs=1e-10)
    assert georaster.affine.c == pytest.approx(expected_affine.c, abs=1e-10)
    assert georaster.affine.d == pytest.approx(expected_affine.d, abs=1e-10)
    assert georaster.affine.e == pytest.approx(expected_affine.e)
    assert georaster.affine.f == pytest.approx(expected_affine.f, abs=1e-10)


def test_rasterization_of_line_simple():
    resolution = 1
    pixels_width = 1

    line = GeoFeature.from_shape(LineString([(2.5, 0), (2.5, 3)]))
    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=5, ymax=5, crs=DEFAULT_CRS)

    fc = FeatureCollection([line])

    expected_image = np.zeros((5, 5), dtype=np.uint8)
    expected_image[2:, 2] = 1

    expected_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)

    expected_crs = DEFAULT_CRS

    expected_result = GeoRaster2(expected_image, expected_affine, expected_crs, nodata=0)

    result = fc.rasterize(resolution, polygonize_width=pixels_width, crs=DEFAULT_CRS, bounds=roi)

    assert result == expected_result


@pytest.mark.parametrize("resolution", [1 / 3, 1 / 6, 1 / 9])
def test_rasterization_of_line_has_correct_pixel_width(resolution):
    xmax, ymax = 11, 5
    pixels_width = 1

    line = GeoFeature.from_shape(LineString([(xmax / 2, 0), (xmax / 2, ymax * 4 / 5)]))
    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=xmax, ymax=ymax, crs=DEFAULT_CRS)

    fc = FeatureCollection([line])

    expected_image = np.zeros((int(ymax // resolution), int(xmax // resolution)), dtype=np.uint8)
    expected_image[int(1 // resolution):, expected_image.shape[1] // 2] = 1

    expected_affine = Affine(resolution, 0.0, 0.0, 0.0, -resolution, 5.0)

    expected_crs = DEFAULT_CRS

    expected_result = GeoRaster2(expected_image, expected_affine, expected_crs, nodata=0)

    result = fc.rasterize(resolution, polygonize_width=pixels_width, crs=DEFAULT_CRS, bounds=roi)

    assert result == expected_result


def test_rasterization_point_single_pixel():
    data = np.zeros((5, 5), dtype=np.uint8)[None, :, :]
    data[0, 2, 2] = 1

    mask = ~(data.astype(bool))

    expected_image = np.ma.masked_array(data, mask)

    fc = FeatureCollection.from_geovectors([
        GeoVector(Point(2.5, 2.5), crs=WEB_MERCATOR_CRS)]
    )

    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=5, ymax=5, crs=WEB_MERCATOR_CRS)

    result = fc.rasterize(1, polygonize_width=1, bounds=roi).image

    assert_array_equal(result.data, expected_image.data)
    assert_array_equal(result.mask, expected_image.mask)


@pytest.mark.parametrize("fill_value,dtype", [
    (1, np.uint8),
    (0, np.uint8),
    (256, np.uint16),
    (1.0, np.float32),
    (1.5, np.float32),
    (0.0, np.float32),
    (256.0, np.float32)
])
def test_rasterization_function_user_dtype(fill_value, dtype):
    resolution = 1

    line = GeoVector.from_bounds(xmin=2, ymin=0, xmax=3, ymax=3, crs=DEFAULT_CRS)
    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=5, ymax=5, crs=DEFAULT_CRS)

    expected_data = np.zeros((5, 5), dtype=dtype)
    expected_data[2:, 2] = fill_value
    expected_mask = np.ones((5, 5), dtype=bool)
    expected_mask[2:, 2] = False
    expected_image = np.ma.masked_array(
        expected_data,
        expected_mask
    )

    expected_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)

    expected_crs = DEFAULT_CRS

    expected_result = GeoRaster2(expected_image, expected_affine, expected_crs)

    result = rasterize([line.get_shape(DEFAULT_CRS)], DEFAULT_CRS, roi.get_shape(DEFAULT_CRS),
                       resolution, fill_value=fill_value, dtype=dtype)

    assert result == expected_result


def test_rasterization_function():
    sq1 = GeoFeature(
        GeoVector.from_bounds(xmin=0, ymin=2, xmax=1, ymax=3, crs=WGS84_CRS),
        {'value': 1.0}
    )
    sq2 = GeoFeature(
        GeoVector.from_bounds(xmin=1, ymin=0, xmax=3, ymax=2, crs=WGS84_CRS),
        {'value': 2.0}
    )
    fc = FeatureCollection([sq1, sq2])

    def func(feat):
        return feat['value']

    expected_image = np.ma.masked_array(
        [[
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 2.0],
            [0.0, 2.0, 2.0]
        ]],
        [
            [False, True, True],
            [True, False, False],
            [True, False, False],
        ]
    )

    result = fc.rasterize(1.0, fill_value=func, crs=WGS84_CRS, dtype=np.float32)

    assert_array_equal(result.image.mask, expected_image.mask)
    assert_array_equal(result.image.data, expected_image.data)


def test_rasterization_function_with_empty_collection():
    fc = FeatureCollection([])

    def func(feat):
        return feat['value']

    expected_image = np.ma.masked_array(
        [[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]],
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )

    bounds = GeoVector.from_bounds(0.0, 0.0, 3.0, 3.0)
    result = fc.rasterize(1.0, bounds=bounds, fill_value=func, crs=WGS84_CRS, dtype=np.float32)

    assert_array_equal(result.image.mask, expected_image.mask)
    assert_array_equal(result.image.data, expected_image.data)


def test_rasterization_function_raises_error_if_no_dtype_is_given():
    sq1 = GeoFeature(
        GeoVector.from_bounds(xmin=0, ymin=2, xmax=1, ymax=3, crs=WGS84_CRS),
        {'value': 1.0}
    )
    sq2 = GeoFeature(
        GeoVector.from_bounds(xmin=1, ymin=0, xmax=3, ymax=2, crs=WGS84_CRS),
        {'value': 2.0}
    )
    fc = FeatureCollection([sq1, sq2])

    with pytest.raises(ValueError) as excinfo:
        fc.rasterize(1.0, fill_value=lambda x: 1, crs=WGS84_CRS)

    assert "dtype must be specified for multivalue rasterization" in excinfo.exconly()
