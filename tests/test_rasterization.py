import pytest

from affine import Affine
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon, LineString, Point

import numpy as np

from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.georaster import GeoRaster2
from telluric.collections import FeatureCollection
from telluric.constants import DEFAULT_CRS, WEB_MERCATOR_CRS

from telluric.rasterization import ScaleError


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
    crs = {'init': 'epsg:32631'}
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

    expected_result = GeoRaster2(expected_image, expected_affine, expected_crs)

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

    expected_result = GeoRaster2(expected_image, expected_affine, expected_crs)

    result = fc.rasterize(resolution, polygonize_width=pixels_width, crs=DEFAULT_CRS, bounds=roi)

    assert result == expected_result


def test_rasterization_point_single_pixel():
    data = np.zeros((5, 5), dtype=np.uint8)[None, :, :]
    data[0, 2, 2] = 1

    mask = ~(data.astype(bool))

    expected_image = np.ma.masked_array(data, mask)

    fc = FeatureCollection.from_geovectors([
        GeoVector(Point(2, 2), crs=WEB_MERCATOR_CRS)]
    )

    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=5, ymax=5, crs=WEB_MERCATOR_CRS)

    result = fc.rasterize(1, polygonize_width=1, bounds=roi).image

    assert_array_equal(result.data, expected_image.data)
    assert_array_equal(result.mask, expected_image.mask)
