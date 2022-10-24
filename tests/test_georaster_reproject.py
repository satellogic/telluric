import pytest
import numpy as np
import rasterio

from telluric import GeoRaster2
from telluric.constants import WGS84_CRS


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject(in_memory):
    """ When called without parameters, output should be same as source """
    raster = GeoRaster2.open("tests/data/raster/rgb.tif", lazy_load=not in_memory)
    expected_raster = raster.reproject()
    assert expected_raster == raster


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_shrink_mask(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif", lazy_load=not in_memory)
    expected_raster = raster.reproject()
    assert not np.isscalar(expected_raster.image.mask)


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject_dimensions(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    expected_raster = raster.reproject(dimensions=(512, 512))
    assert expected_raster.crs == raster.crs
    assert expected_raster.width == 512
    assert expected_raster.height == 512
    assert np.allclose([14.929107, 14.929107],
                       [expected_raster.transform.a, -expected_raster.transform.e])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject_resolution(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    expected_raster = raster.reproject(resolution=30)
    assert expected_raster.crs == raster.crs
    assert expected_raster.width == 255
    assert expected_raster.height == 255
    assert np.allclose([30, 30],
                       [expected_raster.transform.a, -expected_raster.transform.e])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject_bounds(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dst_bounds = [-6574000, -4077000, -6573000, -4076000]
    expected_raster = raster.reproject(dst_bounds=dst_bounds)
    expected_bounds = expected_raster.footprint().get_bounds(expected_raster.crs)
    assert expected_raster.crs == raster.crs
    assert expected_raster.width == 14
    assert expected_raster.height == 14
    assert np.allclose([raster.transform.a, raster.transform.e],
                       [expected_raster.transform.a, expected_raster.transform.e])
    assert np.allclose(expected_bounds[0::3], [-6574000, -4076000])

    # XXX: an extra row and column is produced in the dataset
    # because we are using ceil instead of floor
    # (as rasterio does it internally).
    psize = (expected_raster.transform.a, -expected_raster.transform.e)
    assert np.allclose([expected_bounds[2] - psize[1], expected_bounds[1] + psize[0]],
                       [-6573000, -4077000])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject_bounds_res(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dst_bounds = [-6574000, -4077000, -6570000, -4073000]
    resolution = 30
    expected_raster = raster.reproject(dst_bounds=dst_bounds, resolution=resolution)
    assert expected_raster.crs == raster.crs
    assert expected_raster.width == 134
    assert expected_raster.height == 134
    assert np.allclose(
        expected_raster.footprint().get_bounds(expected_raster.crs),
        dst_bounds)
    assert np.allclose([30, 30],
                       [expected_raster.transform.a, -expected_raster.transform.e])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_no_reproject_src_bounds_dimensions(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    src_bounds = [-6575538, -4078737, -6565840, -4069351]
    dimensions = (1024, 1024)
    expected_raster = raster.reproject(src_bounds=src_bounds, dimensions=dimensions)
    bounds = expected_raster.footprint().get_bounds(raster.crs)
    assert expected_raster.crs == raster.crs
    assert np.allclose(bounds, src_bounds)
    assert np.allclose([9.470703, 9.166015],
                       [expected_raster.transform.a, -expected_raster.transform.e])
    assert expected_raster.width == dimensions[0]
    assert expected_raster.height == dimensions[1]


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_dst_crs(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    expected_raster = raster.reproject(dst_crs=WGS84_CRS)
    assert expected_raster.shape[0] == raster.shape[0]
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == 109
    assert expected_raster.height == 90


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_resolution(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    resolution = 0.01
    expected_raster = raster.reproject(dst_crs=WGS84_CRS, resolution=resolution)
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == 7
    assert expected_raster.height == 6
    assert np.allclose([resolution, resolution],
                       [expected_raster.transform.a, -expected_raster.transform.e])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_dimensions(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dimensions = (100, 100)
    expected_raster = raster.reproject(dst_crs=WGS84_CRS, dimensions=dimensions)
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == dimensions[0]
    assert expected_raster.height == dimensions[1]
    assert np.allclose([0.0006866455078125, 0.0005670066298468868],
                       [expected_raster.transform.a, -expected_raster.transform.e])


@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("bad_params", [
    {'resolution': 10},
    {'dst_bounds': [0, 0, 10, 10]}
])
def test_warp_reproject_dimensions_invalid_params(in_memory, bad_params):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dimensions = (100, 100)
    with pytest.raises(ValueError) as error:
        expected_raster = raster.reproject(
            dst_crs=WGS84_CRS, dimensions=dimensions, **bad_params)
    assert "dimensions cannot be used with" in error.exconly()


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_bounds_no_resolution(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dst_bounds = [-11850000, 4810000, -11849000, 4812000]
    with pytest.raises(ValueError) as error:
        expected_raster = raster.reproject(
            dst_crs=WGS84_CRS, dst_bounds=dst_bounds)
    assert "resolution is required when using src_bounds or dst_bounds" in error.exconly()


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_multi_bounds_fail(in_memory):
    """Mixing dst_bounds and src_bounds fails."""
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dst_bounds = [-11850000, 4810000, -11849000, 4812000]
    with pytest.raises(ValueError) as error:
        expected_raster = raster.reproject(
            dst_crs=WGS84_CRS, dst_bounds=dst_bounds, src_bounds=dst_bounds)
    assert "src_bounds and dst_bounds may not be specified simultaneously" in error.exconly()


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_src_bounds_resolution(in_memory):
    """src_bounds works."""
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    src_bounds = [-6574000, -4077000, -6570000, -4073000]
    resolution = 0.001
    expected_raster = raster.reproject(dst_crs=WGS84_CRS, src_bounds=src_bounds, resolution=resolution)
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == 36
    assert expected_raster.height == 30
    assert np.allclose([resolution, resolution],
                       [expected_raster.transform.a, -expected_raster.transform.e])
    assert np.allclose(expected_raster.footprint().get_bounds(expected_raster.crs),
                       [-59.05524, -34.35851, -59.01924, -34.32851])


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_src_bounds_dimensions(in_memory):
    """src-bounds works with dimensions."""
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    src_bounds = [-6575538, -4078737, -6565840, -4069351]
    dimensions = (1024, 1024)
    expected_raster = raster.reproject(dst_crs=WGS84_CRS, src_bounds=src_bounds, dimensions=dimensions)
    bounds = expected_raster.footprint().get_bounds(expected_raster.crs)
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == dimensions[0]
    assert expected_raster.height == dimensions[1]
    assert np.allclose(bounds[:],
                       [-59.06906, -34.37106, -58.98194, -34.30144])
    assert round(expected_raster.transform.a, 4) == 0.0001
    assert round(-expected_raster.transform.e, 4) == 0.0001


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_reproject_dst_bounds(in_memory):
    """dst_bounds works."""
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    dst_bounds = [-59.05437, -34.35519, -59.01042, -34.32627]
    resolution = 0.001
    expected_raster = raster.reproject(dst_crs=WGS84_CRS, dst_bounds=dst_bounds, resolution=resolution)
    expected_bounds = expected_raster.footprint().get_bounds(expected_raster.crs)
    assert expected_raster.crs == WGS84_CRS
    assert expected_raster.width == 44
    assert expected_raster.height == 29
    assert np.allclose([resolution, resolution],
                       [expected_raster.transform.a, -expected_raster.transform.e])
    assert np.allclose(expected_raster.footprint().get_bounds(expected_raster.crs),
                       dst_bounds)


@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("bad_params", [
    {},
    {'src_bounds': [0, 0, 10, 10], 'resolution': 0.001},
    {'dst_bounds': [0, 0, 10, 10], 'resolution': 0.001},
])
def test_warp_target_aligned_pixels_invalid_params(in_memory, bad_params):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    with pytest.raises(ValueError) as error:
        expected_raster = raster.reproject(
            dst_crs=WGS84_CRS, target_aligned_pixels=True, **bad_params)
    assert "target_aligned_pixels cannot be used" in error.exconly()


@pytest.mark.parametrize("in_memory", [True, False])
def test_warp_target_aligned_pixels_true(in_memory):
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    if in_memory:
        raster_image = raster.image
    resolution = 0.0001
    raster1 = raster.reproject(dst_crs=WGS84_CRS, resolution=resolution,
                               target_aligned_pixels=False)
    raster2 = raster.reproject(dst_crs=WGS84_CRS, resolution=resolution,
                               target_aligned_pixels=True)
    assert raster1 != raster2


def test_warp_reproject_creation_options():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    tiled_raster = raster.reproject(
        dst_crs=WGS84_CRS,
        dimensions=(1024, 1024),
        creation_options={'tiled': True, 'blockxsize': 256, 'blockysize': 256}
    )
    non_tiled_raster = raster.reproject(
        dst_crs=WGS84_CRS,
        dimensions=(1024, 1024),
        creation_options={'tiled': False}
    )
    with rasterio.open(tiled_raster._filename) as src:
        assert src.profile['tiled']
    with rasterio.open(non_tiled_raster._filename) as src:
        assert not src.profile['tiled']
