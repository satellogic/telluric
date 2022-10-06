import pytest
import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from copy import deepcopy

import numpy as np
import rasterio.shutil
from affine import Affine
from rasterio.enums import Resampling, MaskFlags
from unittest.mock import Mock
from PIL import Image
from shapely.geometry import Point, Polygon

from rasterio.crs import CRS
from rasterio.rpc import RPC
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from telluric.constants import WGS84_CRS, WEB_MERCATOR_CRS
from telluric.georaster import GeoRaster2, GeoRaster2Error, GeoRaster2Warning, join, MutableGeoRaster, \
    GeoRaster2IOError
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.collections import FeatureCollection

from common_for_tests import make_test_raster


some_array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
some_mask = np.array([[False, False, False], [False, False, True]], dtype=bool)
some_image_2d = np.ma.array(some_array, mask=some_mask)
some_image_2d_alt = np.ma.array(np.array([[0, 1, 2], [3, 4, 99]], dtype=np.uint8), mask=some_mask)
some_image_3d = np.ma.array(some_array[np.newaxis, :, :], mask=some_mask[np.newaxis, :, :])
some_image_3d_multiband = np.ma.array(
    np.array([some_array, some_array, some_array]), mask=np.array([some_mask, some_mask, some_mask]))
raster_origin = Point(2, 3)
some_affine = Affine.translation(raster_origin.x, raster_origin.y)
some_crs = CRS({'init': 'epsg:32620'})
some_rpcs = RPC(height_off=10.0, height_scale=10.0, lat_off=30.0, lat_scale=30.0,
                line_den_coeff=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                -1.0, -2.0, -3.0, 6.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0],
                line_num_coeff=[2.0, 5.0, 3.0, 4.0, 5.0, 9.0, 7.0, 8.0, 9.0, 10.0, -1.0,
                                -2.0, -3.0, 6.0, 9.0, 2.0, 3.0, 5.0, 6.0, 7.0],
                line_off=0.0, line_scale=0.0, long_off=0.0, long_scale=0.0,
                samp_den_coeff=[4.0, 7.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0,
                                -2.0, -3.0, 7.0, 1.0, 2.0, 7.0, 5.0, 6.0, 7.0],
                samp_num_coeff=[8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                8.0, -2.0, -3.0, 6.0, 1.0, 5.0, 4.0, 5.0, 6.0, 7.0],
                samp_off=50.0, samp_scale=50.0)
some_raster = GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['r'])
some_raster_alt = GeoRaster2(some_image_2d_alt, affine=some_affine, crs=some_crs, band_names=['r'])
some_raster_multiband = GeoRaster2(
    some_image_3d_multiband, band_names=['r', 'g', 'b'], affine=some_affine, crs=some_crs)
default_factors = [2, 4, 8, 16]

some_float32_array = np.array([[[0.0, 0.2], [0.4, 0.6]], [[0.7, 0.8], [0.9, 1.0]]], dtype=np.float32)
some_float32_raster = GeoRaster2(some_float32_array, band_names=[1, 2], affine=some_affine, crs=some_crs, nodata=None)

some_raster_shrunk_mask = some_raster_multiband.copy_with(
    image=np.ma.array(some_raster_multiband.image.data, mask=np.ma.nomask))


def test_construction():
    # test image - different formats yield identical rasters:
    raster_masked_2d = GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, rpcs=some_rpcs)
    raster_masked_3d = GeoRaster2(some_image_3d, affine=some_affine, crs=some_crs, rpcs=some_rpcs)
    raster_masked_array = GeoRaster2(some_array, nodata=5, affine=some_affine, crs=some_crs, rpcs=some_rpcs)
    assert raster_masked_2d == raster_masked_3d
    assert raster_masked_2d == raster_masked_array

    assert np.array_equal(raster_masked_2d.image, some_image_3d)
    assert raster_masked_2d.affine == some_affine
    assert raster_masked_2d.crs == some_crs
    assert raster_masked_2d.rpcs == some_rpcs
    assert raster_masked_2d.dtype == some_image_2d.dtype

    # test bandnames:
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names='gray').band_names == ['gray']
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['gray']).band_names == ['gray']
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs).band_names == [0]
    with pytest.raises(GeoRaster2Error):
        GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['gray', 'red'])


def test_rpcs_init():
    # test rpcs initilization by passing rpcs in different forms
    rpcs_dict = some_rpcs.to_dict()
    rpcs_dict = {k.upper(): v for k, v in rpcs_dict.items()}
    georaster1 = GeoRaster2(some_image_2d, rpcs=some_rpcs)
    georaster2 = GeoRaster2(some_image_2d, rpcs=rpcs_dict)
    assert georaster1.rpcs == georaster2.rpcs
    # rpcs defined as dictionary of str
    new_rpcs_dict = {}
    for k, v in rpcs_dict.items():
        if isinstance(v, float):
            new_rpcs_dict[k] = str(v)
        line_num_coeff = rpcs_dict['LINE_NUM_COEFF']
        new_rpcs_dict['LINE_NUM_COEFF'] = ' '.join(str(e) for e in line_num_coeff)
        line_den_coeff = rpcs_dict['LINE_DEN_COEFF']
        new_rpcs_dict['LINE_DEN_COEFF'] = ' '.join(str(e) for e in line_den_coeff)
        samp_num_coeff = rpcs_dict['SAMP_NUM_COEFF']
        new_rpcs_dict['SAMP_NUM_COEFF'] = ' '.join(str(e) for e in samp_num_coeff)
        samp_den_coeff = rpcs_dict['SAMP_DEN_COEFF']
        new_rpcs_dict['SAMP_DEN_COEFF'] = ' '.join(str(e) for e in samp_den_coeff)
    georaster3 = GeoRaster2(some_image_2d, rpcs=new_rpcs_dict)
    assert georaster2.rpcs == georaster3.rpcs


def test_eq():
    """ test ._eq_ """
    assert some_raster == some_raster
    assert some_raster != some_raster.copy_with(image=some_raster.image + 1)
    assert some_raster != some_raster.copy_with(affine=Affine.translation(42, 42))
    assert some_raster != some_raster.copy_with(crs={'init': 'epsg:32621'})

    # np.ma.nomask
    assert (
        some_raster.copy_with(image=np.ma.masked_array(
            some_raster.image.data,
            mask=np.ma.nomask)) ==
        some_raster.copy_with(image=np.ma.masked_array(
            some_raster.image.data,
            mask=np.zeros_like(some_raster.image.data, dtype=bool))))
    assert (
        some_raster.copy_with(image=np.ma.masked_array(
            some_raster.image.data,
            mask=np.zeros_like(some_raster.image.data, dtype=bool))) ==
        some_raster.copy_with(image=np.ma.masked_array(
            some_raster.image.data,
            mask=np.ma.nomask)))


def test_copy():
    assert some_raster == some_raster.copy()
    assert isinstance(some_raster.copy(mutable=True), MutableGeoRaster)
    raster = GeoRaster2.open('./tests/data/raster/overlap1.tif')
    assert raster == raster.copy()
    # assert raster.copy().not_loaded()
    raster.image
    assert raster == raster.copy()
    assert not raster.copy().not_loaded()


def test_eq_ignores_masked_values():
    assert some_raster == some_raster_alt


def test_read_write():
    for extension in ['tif', 'png']:
        with TemporaryDirectory() as folder:
            path = os.path.join(folder, 'test.%s' % extension)
            some_raster_multiband.save(path, factors=default_factors)
            read = GeoRaster2.open(path)
            assert read == some_raster_multiband


def test_read_non_georeferenced(recwarn):
    crs = CRS(init='epsg:3857')
    affine = Affine(10.0, 0.0, -6425941.63996855,
                    0.0, -10.0, -3169315.69478084)
    raster = GeoRaster2.open('tests/data/raster/no_georef.png', crs=crs, affine=affine, lazy_load=False)
    assert raster.crs == crs
    assert raster.affine == affine


def test_read_write_internal_external_mask():
    with TemporaryDirectory() as folder:
        # internal mask (default) leaves no .msk file:
        internal_path = os.path.join(folder, 'internal.tif')
        some_raster_multiband.save(internal_path, factors=default_factors)
        assert not os.path.exists(internal_path + '.msk')

        # external mask leaves .msk file:
        external_path = os.path.join(folder, 'external.tif')
        some_raster_multiband.save(external_path, GDAL_TIFF_INTERNAL_MASK=False, factors=default_factors)
        assert os.path.exists(external_path + '.msk')

        # other than that, both rasters are identical:
        assert GeoRaster2.open(internal_path) == GeoRaster2.open(external_path)


def test_read_respects_nodata():
    expected_nodata = 100
    path = '/vsimem/raster_for_test.tif'
    some_raster.save(path, nodata=expected_nodata)

    assert expected_nodata == GeoRaster2.open(path).nodata_value


def test_save_preserves_nodata():
    expected_nodata = 200
    path = '/vsimem/raster_for_test.tif'
    raster_nodata = GeoRaster2(some_array, nodata=expected_nodata, affine=some_affine, crs=some_crs)
    raster_nodata.save(path)
    assert expected_nodata == GeoRaster2.open(path).nodata_value


def test_tags():
    with TemporaryDirectory() as folder:
        path = os.path.join(folder, 'test.tif')
        some_raster_multiband.save(path, tags={'foo': 'bar'}, factors=default_factors)

        assert GeoRaster2.tags(path) == {'AREA_OR_POINT': 'Area', 'foo': 'bar',
                                         'telluric_band_names': '["r", "g", "b"]'}  # namespace=default
        assert GeoRaster2.tags(path, 'IMAGE_STRUCTURE') == {'COMPRESSION': 'LZW', 'INTERLEAVE': 'PIXEL'}


def test_deepcopy():
    """ Tests .__copy__() and .__deepcopy__() """
    a_raster = some_raster.deepcopy_with()
    deep_copy = deepcopy(a_raster)
    a_raster.image.setflags(write=1)
    a_raster.image.data[0, 0, 0] += 1
    a_raster.image.setflags(write=0)
    assert deep_copy.image[0, 0, 0] == a_raster.image[0, 0, 0] - 1


def test_copy_with():
    new_affine = Affine.translation(42, 42)
    assert some_raster.copy_with(affine=new_affine).affine == new_affine


@pytest.mark.parametrize("raster", [some_raster, some_raster.save("/vsimem/test_resize.tif")])
def test_resize(raster):
    resampling_modes = [m.name for m in Resampling]
    for resampling_name in resampling_modes:
        resampling = Resampling[resampling_name]
        print('\nresampling name:', resampling_name)
        if resampling_name in ['min', 'max', 'med', 'q1', 'q3', 'sum']:
            print('\nskipping', resampling_name)
            continue
        if resampling_name not in ['bilinear', 'cubic', 'cubic_spline', 'lanczos', 'gauss']:
            resized_raster = raster.resize(2, resampling=resampling).resize(.5, resampling=resampling)
            assert (raster.image == resized_raster.image).all()
            assert raster == resized_raster
            # continue
        if resampling_name not in ['cubic_spline']:
            assert raster.resize(ratio=1, resampling=resampling) == some_raster

        assert raster.resize(ratio=2, resampling=resampling).width == 2 * some_raster.width
        assert raster.resize(ratio=2, resampling=resampling).shape == (1, 4, 6)
        assert raster.resize(dest_height=42, resampling=resampling).height == 42
        assert raster.resize(dest_width=42, resampling=resampling).width == 42
        assert raster.resize(dest_width=42, dest_height=42, resampling=resampling).width == 42
        assert raster.resize(dest_width=42, dest_height=42, resampling=resampling).height == 42
        assert raster.resize(dest_resolution=42, resampling=resampling).resolution() == 42

    with pytest.raises(GeoRaster2Error):
        raster.resize(ratio=1, dest_width=2)
    with pytest.raises(GeoRaster2Error):
        raster.resize(ratio_x=2)


def test_to_pillow_image():
    # without mask:
    img = some_raster_multiband.to_pillow_image()
    assert img.height == some_raster_multiband.height
    assert img.width == some_raster_multiband.width
    assert len(img.getbands()) == some_raster_multiband.num_bands

    # with mask:
    img, mask = some_raster_multiband.to_pillow_image(return_mask=True)
    assert mask.height == some_raster_multiband.height
    assert mask.width == some_raster_multiband.width
    assert len(mask.getbands()) == 1

    # with shrunk mask:
    img, mask = some_raster_shrunk_mask.to_pillow_image(return_mask=True)
    assert mask.height == some_raster_multiband.height
    assert mask.width == some_raster_multiband.width
    assert len(mask.getbands()) == 1


def test_num_pixels():
    assert some_raster_multiband.num_pixels() == 6
    assert some_raster_multiband.num_pixels_data() == 5
    assert some_raster_multiband.num_pixels_nodata() == 1


def test_limit_to_bands():
    with pytest.raises(GeoRaster2Error) as error:
        some_raster_multiband.limit_to_bands(['not-existing'])
    assert "requested bands {'not-existing'} that are not found in raster" in error.exconly()

    bands = ['g', 'b']
    selected = some_raster_multiband.limit_to_bands(bands)
    assert selected.band_names == bands


def test_limit_to_bands_off_memory():
    r1 = GeoRaster2.open("tests/data/raster/rgb.tif", band_names=['r', 'g', 'b'], lazy_load=False)
    r2 = GeoRaster2.open("tests/data/raster/rgb.tif", band_names=['r', 'g', 'b'])
    assert r1.limit_to_bands(['b', 'r']) == r2.limit_to_bands(['b', 'r'])
    assert r1.limit_to_bands(['r', 'b']) != r2.limit_to_bands(['b', 'r'])
    assert r2._image is None


def test_to_png_singleband():
    pytest.importorskip("matplotlib")
    raster = some_raster
    png_bytes = raster.colorize("gray").to_png(transparent=True, thumbnail_size=512)
    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    assert img.size == raster.to_pillow_image().size


def test_to_png_multiband():
    raster = some_raster_multiband
    png_bytes = raster.to_png(transparent=True, thumbnail_size=512)
    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    assert img.size == raster.to_pillow_image().size


def test_to_png_uint16(recwarn):
    raster = make_test_raster(257 * 42, band_names=[1, 2, 3], dtype=np.uint16)

    png_bytes = raster.to_png(transparent=True)
    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "downscaling dtype to 'uint8' to convert to png"

    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    expected_image_size = raster.astype(np.uint8).to_pillow_image().size
    assert img.size == expected_image_size
    assert raster.astype(np.uint8) == GeoRaster2.from_bytes(png_bytes, affine=raster.affine,
                                                            crs=raster.crs, band_names=raster.band_names)


def test_to_png_int32(recwarn):
    raster = make_test_raster(257 * 42, band_names=[1, 2, 3], dtype=np.int32)

    png_bytes = raster.to_png(transparent=True)
    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "downscaling dtype to 'uint8' to convert to png"

    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    expected_image_size = raster.astype(np.uint8).to_pillow_image().size
    assert img.size == expected_image_size
    assert raster.astype(np.uint8) == GeoRaster2.from_bytes(png_bytes, affine=raster.affine,
                                                            crs=raster.crs, band_names=raster.band_names)


def test_to_png_from_bytes():
    arr = np.array([np.full((3, 5), 1), np.full((3, 5), 5), np.full((3, 5), 10)], dtype=np.uint8)
    raster = GeoRaster2(image=arr, affine=Affine.identity(), crs=WEB_MERCATOR_CRS, band_names=['r', 'g', 'b'])
    png_bytes = raster.to_png()
    assert raster == GeoRaster2.from_bytes(png_bytes, affine=raster.affine,
                                           crs=raster.crs, band_names=raster.band_names)


def test_to_png_uses_the_first_band_for_a_two_bands_raster(recwarn):
    raster = make_test_raster(257 * 42, band_names=[1, 2], dtype=np.int32)

    png_bytes = raster.to_png(transparent=True, thumbnail_size=512)
    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "Deprecation: to_png of less then three bands raster will be not be supported in next \
release, please use: .colorize('gray').to_png()"
    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "Limiting two bands raster to use the first band to generate png"

    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "downscaling dtype to 'uint8' to convert to png"

    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    expected_image_size = raster.limit_to_bands([1]).astype(np.uint8).to_pillow_image().size
    assert img.size == expected_image_size


def test_to_png_uses_warns_on_single_bands_raster(recwarn):
    raster = make_test_raster(42, band_names=[1], dtype=np.uint8)

    png_bytes = raster.to_png(transparent=True, thumbnail_size=512)
    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "Deprecation: to_png of less then three bands raster will be not be supported in next \
release, please use: .colorize('gray').to_png()"
    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    expected_image_size = raster.limit_to_bands([1]).astype(np.uint8).to_pillow_image().size
    assert img.size == expected_image_size


def test_to_png_uses_the_first_three_bands_for_a_more_than_three_bands_raster(recwarn):
    raster = make_test_raster(257 * 42, band_names=[1, 2, 3, 4], dtype=np.int32)

    png_bytes = raster.to_png(transparent=True, thumbnail_size=512)

    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "Limiting %d bands raster to first three bands to generate png" % raster.num_bands

    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "downscaling dtype to 'uint8' to convert to png"

    img = Image.frombytes('RGBA', (raster.width, raster.height), png_bytes)
    expected_image_size = raster.limit_to_bands([1, 2]).astype(np.uint8).to_pillow_image().size
    assert img.size == expected_image_size


def test_to_raster_to_world():
    pt = Point(1, 1)
    in_world = some_raster.to_world(pt)
    assert pytest.approx(in_world.get_shape(in_world.crs).x) == raster_origin.x + pt.x
    assert pytest.approx(in_world.get_shape(in_world.crs).y) == raster_origin.y + pt.y

    back_in_image = some_raster.to_raster(in_world)
    assert back_in_image.equals_exact(pt, tolerance=5e-07)


def test_corner_invalid():
    with pytest.raises(GeoRaster2Error) as error:
        some_raster.corner('foo')
    assert '%s' % error.value == "corner foo invalid, expected: ['ul', 'ur', 'br', 'bl']"


def test_corner():
    expected_image_corners = {
        'ul': Point(0, 0),
        'ur': Point(some_raster.width, 0),
        'bl': Point(0, some_raster.height),
        'br': Point(some_raster.width, some_raster.height),
    }
    expected_corners = {
        'ul': Point(raster_origin.x + 0, raster_origin.y + 0),
        'ur': Point(raster_origin.x + some_raster.width, raster_origin.y + 0),
        'bl': Point(raster_origin.x + 0, raster_origin.y + some_raster.height),
        'br': Point(raster_origin.x + some_raster.width, raster_origin.y + some_raster.height),
    }

    for corner in GeoRaster2.corner_types():
        assert some_raster.corner(corner).almost_equals(GeoVector(expected_corners[corner], some_raster.crs))
        assert some_raster.image_corner(corner).equals_exact(expected_image_corners[corner], tolerance=5e-07)


def test_center():
    ul = some_raster.corner('ul').get_shape(some_raster.crs)
    br = some_raster.corner('br').get_shape(some_raster.crs)
    expected_center = Point((ul.x + br.x) / 2, (ul.y + br.y) / 2)
    expected_center_vector = GeoVector(expected_center, some_raster.crs)

    assert expected_center_vector.equals_exact(some_raster.center(), tolerance=5e-07)


def test_bounds():
    expected = Polygon([[0, 0],
                        [some_raster.width, 0],
                        [some_raster.width, some_raster.height],
                        [0, some_raster.height]])

    assert some_raster.bounds().equals_exact(expected, tolerance=5e-07)


def test_footprint():
    expected_shp = Polygon([[raster_origin.x + 0, raster_origin.y + 0],
                            [raster_origin.x + some_raster.width, raster_origin.y + 0],
                            [raster_origin.x + some_raster.width, raster_origin.y + some_raster.height],
                            [raster_origin.x + 0, raster_origin.y + some_raster.height]])
    expected = GeoVector(expected_shp, some_raster.crs)

    assert some_raster.footprint().almost_equals(expected)


def test_area():
    scale = 2
    raster = some_raster.copy_with(affine=some_raster.affine * Affine.scale(scale))
    expected = raster.width * raster.height * (scale ** 2)
    assert pytest.approx(expected, .01) == raster.area()


def test_reduce():
    for op in ['min', 'max', 'sum', 'mean', 'var', 'std']:
        expected = getattr(np, op)([0, 1, 2, 3, 4])
        actual = getattr(some_raster, op)()[0]
        assert expected == actual


def test_histogram():
    hist = some_raster.histogram()
    assert np.count_nonzero(hist['r']) == 5
    assert hist.length == 256


def test_invert():
    assert ((~some_raster).image.mask != some_raster.image.mask).all()
    assert ~~some_raster == some_raster


def test_mask():
    image = np.ma.masked_array(np.random.uniform(size=(3, 4, 5)).astype(np.uint8), np.full((3, 4, 5), False))
    raster = some_raster.deepcopy_with(image=image, band_names=['r', 'g', 'b'])
    w, h = raster.width // 2, raster.height // 2
    left_quadrant = Polygon([[0, 0], [0, h], [w, h], [w, 0]])
    vector = raster.to_world(left_quadrant, dst_crs=raster.crs)

    masked = raster.mask(vector)
    assert not masked.image[:, :w, :h].mask.any()
    assert masked.image[:, w:, h:].mask.all()

    # masking by GeoFeature
    feature = GeoFeature(vector, properties={'prop1': 1, 'prop2': 2})
    masked = raster.mask(feature)
    assert not masked.image[:, :w, :h].mask.any()
    assert masked.image[:, w:, h:].mask.all()

    # raster masked "inside shape" should be inverse to raster masked "outside shape":
    assert raster.mask(vector, mask_shape_nodata=True) == ~raster.mask(vector)


def test_mask_by_value():
    expected_mask = np.array([
        [[False, True, False], [False, False, False]],
        [[False, True, False], [False, False, False]],
        [[False, True, False], [False, False, False]]], dtype=bool)

    assert np.array_equal(expected_mask, some_raster_multiband.mask_by_value(1).image.mask)


def test_get_item():
    raster = some_raster.resize(2)
    assert raster[:, :] == raster

    assert (raster[1:-1, :].width, raster[1:-1, :].height) == (raster.width - 2, raster.height)
    assert (raster[:, 1:-1].width, raster[:, 1:-1].height) == (raster.width, raster.height - 2)

    assert raster[1:, 1:].corner('br').almost_equals(raster.corner('br'))

    with pytest.raises(GeoRaster2Error):
        raster[1:-1]


def test_slicing_negative_bounds_raises_warning_and_cutsoff_by_zero(recwarn):
    negative_slice = some_raster[-1:, -1:]

    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "Negative indices are not supported and were rounded to zero"

    assert negative_slice == some_raster


def test_resolution():
    raster = some_raster.deepcopy_with(affine=Affine.scale(2, 3))
    assert raster.resolution() == np.sqrt(2 * 3)


def test_empty_from_roi():
    roi = GeoVector(
        Polygon.from_bounds(12.36, 42.05, 12.43, 42.10),
        WGS84_CRS
    ).reproject(WEB_MERCATOR_CRS)
    resolution = 20.0
    band_names = ["a", "b", "c"]
    some_dtype = np.uint16

    empty = GeoRaster2.empty_from_roi(roi, resolution, band_names, some_dtype)

    # Cannot compare approximate equality of polygons because the
    # topology might be different, see https://github.com/Toblerity/Shapely/issues/535
    # (Getting the envelope seems to fix the problem)
    # Also, reprojecting the empty footprint to WGS84 permits using
    # a positive number of decimals (the relative error is of course
    # the same)
    assert empty.footprint().reproject(WGS84_CRS).envelope.almost_equals(roi.envelope, decimal=3)
    assert empty.resolution() == resolution
    assert empty.crs == roi.crs
    assert empty.band_names == band_names
    assert empty.dtype == some_dtype
    assert empty.affine.determinant == -1 * resolution * resolution


def test_georaster_contains_geometry():
    roi = GeoVector(
        Polygon.from_bounds(12.36, 42.05, 12.43, 42.10),
        WGS84_CRS
    ).reproject(WEB_MERCATOR_CRS)
    resolution = 20.0

    empty = GeoRaster2.empty_from_roi(roi, resolution)

    assert roi in empty
    assert roi.buffer(-1) in empty
    assert roi.buffer(1) not in empty


def test_empty_from_roi_respects_footprint():
    # See https://github.com/satellogic/telluric/issues/39
    raster = GeoRaster2.open("tests/data/raster/overlap1.tif")

    empty = GeoRaster2.empty_from_roi(shape=raster.shape[1:][::-1], ul_corner=(v[0] for v in raster.corner('ul').xy),
                                      resolution=raster.res_xy(), crs=raster.crs,
                                      band_names=raster.band_names, dtype=raster.dtype)

    empty_simple = GeoRaster2.empty_from_roi(roi=raster.footprint(),
                                             resolution=raster.res_xy(),
                                             band_names=raster.band_names, dtype=raster.dtype)

    assert raster.footprint().almost_equals(empty.footprint())
    assert raster.footprint().almost_equals(empty_simple.footprint())


def test_astype_uint8_to_uint8_conversion():
    raster_uint8 = make_test_raster(value=42, band_names=[1, 2], dtype=np.uint8)
    raster_uint8_copy = raster_uint8.astype(np.uint8)
    assert raster_uint8 == raster_uint8_copy


def test_astype_uint8_to_uint16_conversion():
    raster_uint8 = make_test_raster(value=42, band_names=[1, 2], dtype=np.uint8)
    raster_uint16 = raster_uint8.astype(np.uint16)
    expected_raster_uint16 = make_test_raster(257 * 42, band_names=[1, 2], dtype=np.uint16)
    assert raster_uint16 == expected_raster_uint16


def test_astype_uint16_to_uint8_conversion():
    expected_raster_uint8 = make_test_raster(value=20, band_names=[1, 2], dtype=np.uint8)
    raster_uint16 = make_test_raster(128 * 42, band_names=[1, 2], dtype=np.uint16)
    raster_uint8 = raster_uint16.astype(np.uint8)
    assert raster_uint8 == expected_raster_uint8


def test_astype_uint8_to_uint8_roundtrip():
    raster_uint8 = make_test_raster(value=20, band_names=[1, 2], dtype=np.uint8)
    assert raster_uint8 == raster_uint8.astype(np.uint16).astype(np.uint8)


def test_astype_uint8_to_int32_conversion():
    raster_uint8 = make_test_raster(42, band_names=[1, 2], dtype=np.uint8)
    raster_int32 = raster_uint8.astype(np.int32)
    expected_value = (2**32 - 1) / (2**8 - 1) * (42 - 127.5) - 1
    expected_raster_int32 = make_test_raster(expected_value, band_names=[1, 2], dtype=np.int32)
    assert raster_int32 == expected_raster_int32


def test_astype_raises_error_for_missing_ranges():
    raster = make_test_raster(value=20, band_names=[1, 2], dtype=np.uint8)
    with pytest.raises(GeoRaster2Error) as excinfo:
        raster.astype(np.uint8, in_range=None, out_range='dtype')

    assert "Both ranges should be specified or none of them." in excinfo.exconly()

    with pytest.raises(GeoRaster2Error) as excinfo:
        raster.astype(np.uint8, in_range='dtype', out_range=None)

    assert "Both ranges should be specified or none of them." in excinfo.exconly()


def test_astype_raises_error_for_dtype_out_range_for_non_integer():
    raster = make_test_raster(value=20, band_names=[1, 2], dtype=np.uint8)
    with pytest.raises(GeoRaster2Error) as excinfo:
        raster.astype(np.float32, out_range='dtype')

    assert "Value 'dtype' of out_range parameter is supported only for integer type." in excinfo.exconly()


def test_astype_float32_to_uint8_conversion():
    with pytest.warns(GeoRaster2Warning):
        raster_uint8 = some_float32_raster.astype(np.uint8)
    expected_raster_uint8 = GeoRaster2(image=np.array([
        [[0, 51], [102, 153]],
        [[178, 204], [229, 255]]], dtype=np.uint8),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_uint8_conversion_with_out_range():
    with pytest.warns(GeoRaster2Warning):
        raster_uint8 = some_float32_raster.astype(np.uint8, out_range=(100, 200))
    expected_raster_uint8 = GeoRaster2(image=np.array([
        [[100, 120], [140, 160]],
        [[169, 180], [189, 200]]], dtype=np.uint8),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_uint8_conversion_with_in_range():
    raster_uint8 = some_float32_raster.astype(np.uint8, in_range=(0.5, 1.0))
    expected_raster_uint8 = GeoRaster2(image=np.array([
        [[0, 0], [0, 51]],
        [[101, 153], [203, 255]]], dtype=np.uint8),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_uint8_conversion_without_stretching():
    raster_float32 = make_test_raster(10.0, band_names=[1, 2], dtype=np.float32)
    raster_uint8 = raster_float32.astype(np.uint8, in_range=None, out_range=None)
    expected_raster_uint8 = make_test_raster(10, band_names=[1, 2], dtype=np.uint8)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_int8_conversion():
    with pytest.warns(GeoRaster2Warning):
        raster_uint8 = some_float32_raster.astype(np.int8)
    expected_raster_uint8 = GeoRaster2(image=np.array([
        [[-128, -76], [-25, 25]],
        [[50, 76], [101, 127]]], dtype=np.int8),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_int8_conversion_with_clip_negative():
    with pytest.warns(GeoRaster2Warning):
        raster_uint8 = some_float32_raster.astype(np.int8, clip_negative=True)
    expected_raster_uint8 = GeoRaster2(image=np.array([
        [[0, 25], [50, 76]],
        [[88, 101], [114, 127]]], dtype=np.int8),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_uint8 == expected_raster_uint8


def test_astype_float32_to_float16_conversion():
    with pytest.warns(GeoRaster2Warning):
        raster_float16 = some_float32_raster.astype(np.float16, out_range=(0.0, 10.0))
    expected_raster_float16 = GeoRaster2(image=np.array([
        [[0.0, 2.0], [4.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0]]], dtype=np.float16),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float16 == expected_raster_float16


def test_astype_float32_to_float16_conversion_without_stretching():
    raster_float16 = some_float32_raster.astype(np.float16, in_range=None, out_range=None)
    expected_raster_float16 = GeoRaster2(
        some_float32_array.astype(np.float16),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float16 == expected_raster_float16


def test_astype_uint8_to_float32_conversion():
    raster_float32 = some_raster.astype(np.float32, out_range=(0, 1))
    expected_raster_float32 = GeoRaster2(image=np.ma.array(
        np.array([
            [0.0, 0.003921568859368563, 0.007843137718737125],
            [0.0117647061124444, 0.01568627543747425, 1.0]
        ], dtype=np.float32), mask=some_raster.image.mask),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float32 == expected_raster_float32


def test_astype_uint8_to_float32_conversion_with_image_in_range():
    raster_float32 = some_raster.astype(np.float32, in_range='image', out_range=(0, 1))
    expected_raster_float32 = GeoRaster2(image=np.ma.array(
        np.array([
            [0.0, 0.25, 0.5],
            [0.75, 1.0, 1.0]
        ], dtype=np.float32), mask=some_raster.image.mask),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float32 == expected_raster_float32


def test_astype_uint8_to_float32_conversion_with_custom_in_range():
    raster_float32 = some_raster.astype(np.float32, in_range=('min', 'max'), out_range=(0, 1))
    expected_raster_float32 = GeoRaster2(image=np.ma.array(
        np.array([
            [0.0, 0.25, 0.5],
            [0.75, 1.0, 1.0]
        ], dtype=np.float32), mask=some_raster.image.mask),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float32 == expected_raster_float32

    raster_float32 = some_raster.astype(np.float32, in_range=(2, 4), out_range=(0, 1))
    expected_raster_float32 = GeoRaster2(image=np.ma.array(
        np.array([
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 1.0]
        ], dtype=np.float32), mask=some_raster.image.mask),
        affine=some_float32_raster.affine, crs=some_float32_raster.crs, nodata=None)
    assert raster_float32 == expected_raster_float32


def test_astype_uint8_to_float32_conversion_without_stretching():
    raster_uint8 = make_test_raster(value=20, band_names=[1, 2], dtype=np.uint8)
    raster_float32 = raster_uint8.astype(np.float32, in_range=None, out_range=None)
    expected_raster_float32 = make_test_raster(value=20, band_names=[1, 2], dtype=np.float32)
    assert raster_float32 == expected_raster_float32


def test_png_thumbnail_has_expected_properties():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    expected_thumbnail = raster.resize(dest_width=512, resampling=Resampling.nearest)
    result_thumbnail = GeoRaster2.from_bytes(
        raster.to_png(transparent=True, thumbnail_size=512, resampling=Resampling.nearest, in_range='image'),
        affine=expected_thumbnail.affine, crs=expected_thumbnail.crs, band_names=expected_thumbnail.band_names
    )

    assert result_thumbnail == expected_thumbnail


def test_destructor():
    with NamedTemporaryFile(suffix='.tif', delete=False) as src:
        raster = GeoRaster2(filename=src.name, temporary=True)
        raster = some_raster
        assert not os.path.isfile(src.name)


def test_save_temporary():
    with NamedTemporaryFile(suffix='.tif', delete=False) as src, NamedTemporaryFile(suffix='.tif') as dst:
        # create valid raster file
        some_raster.save(src.name)
        raster = GeoRaster2(filename=src.name, temporary=True)
        assert raster._filename == src.name
        assert raster._temporary

        raster.save(dst.name)
        assert not raster._temporary
        assert raster._filename is None
        # temporary file is removed
        assert not os.path.isfile(src.name)


def test_save_uses_copy():
    band_names = ["b1", "b2", "b3"]
    raster = GeoRaster2.open("tests/data/raster/rgb.tif", band_names=band_names)
    with NamedTemporaryFile(suffix='.tif') as target:
        factors = [2, 4]
        result = raster.save(target.name, overviews=True, factors=factors)
        assert result.band_names == band_names
        assert result.overviews_factors == factors


def test_reproject_lazy():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    reprojected = raster.reproject(dst_crs=WGS84_CRS)
    assert reprojected._image is None
    assert reprojected._filename is not None
    assert reprojected._temporary


def test_reproject_rpcs():
    # test reprojection when rpcs are defined and input raster has no src_crs
    # TODO: add smaller test data
    raster = GeoRaster2.open("tests/data/raster/grayscale.tif")
    reprojected = raster.reproject(dst_crs=WEB_MERCATOR_CRS,
                                   rpcs=raster.rpcs)
    assert reprojected.shape == (1, 2072, 5241)
    assert reprojected.mean()[0] == pytest.approx(724.4861459505134, 1e-4)


def test_no_error_empty_rpcs():
    # Make sure that calling rpcs method doesn't raise an exception
    # if the underlying file was deleted and the instance was created
    # with lazy_load=False.
    path = "/vsimem/raster_for_test.tif"
    some_raster.save(path)
    raster = GeoRaster2.open(path, lazy_load=False)
    assert raster.rpcs is None

    rasterio.shutil.delete(path)
    assert not rasterio.shutil.exists(path)
    assert raster.rpcs is None


def test_reproject_when_having_no_data_values_in_image():
    shape = (1, 500, 500)
    arr = np.ones(shape, dtype=np.float32)
    for i in range(500):
        for j in range(500):
            if j % 2 == 0:
                arr[0, i, j] = 300
    arr = np.ma.masked_array(arr, arr == 1)
    raster = make_test_raster(image=arr)
    reprojected_raster = raster.reproject(WGS84_CRS, resampling=Resampling.average)
    counts = np.unique(reprojected_raster.image, return_counts=True)
    assert counts[0] == 300
    assert len(counts) == 2


def test_georaster_save_unity_affine_emits_warning(recwarn):
    shape = (1, 500, 500)
    arr = np.ones(shape)
    aff = Affine.scale(1, -1)
    raster = GeoRaster2(image=arr, affine=aff, crs=WEB_MERCATOR_CRS)

    with NamedTemporaryFile(suffix=".tif") as fp:
        raster.save(fp.name)

    w = recwarn.pop(NotGeoreferencedWarning)
    assert "The given matrix is equal to Affine.identity or its flipped counterpart." in str(w.message)


def test_georaster_save_emits_warning_if_uneven_mask(recwarn):
    affine = Affine.translation(0, 2) * Affine.scale(1, -1)
    raster = GeoRaster2(
        image=np.array([
            [
                [100, 200],
                [100, 200]
            ],
            [
                [110, 0],
                [110, 0]
            ]
        ], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        nodata=0
    )

    orig_mask = raster.image.mask

    assert not (orig_mask == orig_mask[0]).all()

    with NamedTemporaryFile(suffix=".tif") as fp:
        raster.save(fp.name)

    w = recwarn.pop(GeoRaster2Warning)
    assert (
        "Saving different masks per band is not supported, the union of the masked values will be performed."
        in str(w.message)
    )


@pytest.mark.parametrize("bounds", (
    (-6572853, -4072334, -6569253, -4068359),  # roi intersects raster's bounds
    (-6572853, -4075105, -6569253, -4071131),  # roi lies inside raster's bounds
    (-6565154, -4075105, -6561555, -4071131),  # roi doesn't intersect raster's bounds
))
def test_crop_boundless_masked(bounds):
    raster_w_mask = GeoRaster2.open("tests/data/raster/rgb.tif")
    raster_wo_mask = GeoRaster2.open("tests/data/raster/rgb.jp2")

    roi = GeoVector(Polygon.from_bounds(*bounds), WEB_MERCATOR_CRS)
    assert np.array_equal(
        raster_w_mask.crop(roi).image.mask, raster_wo_mask.crop(roi).image.mask
    )

    feature = GeoFeature(roi, properties={"prop1": 1, "prop2": 2})
    assert np.array_equal(
        raster_w_mask.crop(feature).image.mask, raster_wo_mask.crop(feature).image.mask
    )

    collection = FeatureCollection([feature])
    assert np.array_equal(
        raster_w_mask.crop(collection).image.mask,
        raster_wo_mask.crop(collection).image.mask,
    )


def test_crop_respects_rounding_precision():
    # https://github.com/satellogic/telluric/pull/311
    r1 = GeoRaster2.open("tests/data/raster/non_aligned1.tif", lazy_load=False)
    r2 = GeoRaster2.open("tests/data/raster/non_aligned2.tif", lazy_load=False)
    roi = GeoVector.from_bounds(
        271806.0179640717,
        1438839.4164977344,
        337216.40041283204,
        1519170.2272236191,
        crs=r1.crs,
    )
    r1_cropped = r1.crop(roi)
    r2_cropped = r2.crop(roi)
    assert r1.shape == r2.shape
    assert r1_cropped.shape == r2_cropped.shape


def test_gdal_open_env_when_proxy_is_not_set_and_url_not_with_http():
    url = "https://bla.com"
    envs = GeoRaster2.get_gdal_env(url)
    assert envs == {}


def test_gdal_open_env_when_proxy_is_not_set_and_url_with_http():
    url = "http://bla.com"
    envs = GeoRaster2.get_gdal_env(url)
    assert envs == {}


def test_gdal_open_env_when_proxy_is_set_and_url_with_http(monkeypatch):
    monkeypatch.setitem(os.environ, "TELLURIC_HTTP_PROXY", "http://proxy")
    url = "http://bla.com"
    envs = GeoRaster2.get_gdal_env(url)
    assert envs == {'GDAL_HTTP_PROXY': 'http://proxy'}


def test_gdal_open_env_when_proxy_is_set_and_url_with_https(monkeypatch):
    monkeypatch.setitem(os.environ, "TELLURIC_HTTP_PROXY", "http://proxy")
    url = "https://bla.com"
    envs = GeoRaster2.get_gdal_env(url)
    assert envs == {}


def test_useses_mask_when_there_is_mask():
    raster = Mock()
    raster.mask_flag_enums = ([MaskFlags.per_dataset], [MaskFlags.per_dataset])
    assert GeoRaster2._read_with_mask(raster, masked=None)


def test_doesnt_use_mask_when_override_mask():
    raster = Mock()
    raster.mask_flag_enums = ([MaskFlags.per_dataset], [MaskFlags.per_dataset])
    assert not GeoRaster2._read_with_mask(raster, masked=False)


def test_doesnt_use_mask_when_no_mask():
    raster = Mock()
    raster.mask_flag_enums = ([MaskFlags.nodata], [MaskFlags.per_dataset])
    assert not GeoRaster2._read_with_mask(raster, masked=None)


def rasters_for_testing_chunks():
    rasters = [GeoRaster2.open("tests/data/raster/overlap2.tif"), GeoRaster2.open("tests/data/raster/overlap2.tif")]
    rasters[0].image
    return rasters


def test_join_use_mask_band():
    # https://github.com/OSGeo/gdal/issues/1148
    # The test checks the pixel at the coordinate of bottom left corner of the r2 raster is unmasked.
    r1 = GeoRaster2.open("tests/data/raster/overlap1.tif")
    r2 = GeoRaster2.open("tests/data/raster/overlap2.tif")
    joined = join([r1, r2])
    assert not joined.get(r2.corners()["bl"]).mask.all()


@pytest.mark.parametrize("raster", rasters_for_testing_chunks())
def test_chunks_with_pad(raster):
    shape = (256, 256)

    # validate that each chunk is in the right offsets
    for r, offsets in raster.chunks(shape, pad=True):
        expected_raster = raster.get_window(Window(col_off=offsets[0], row_off=offsets[
                                            1], width=shape[0], height=shape[1]))
        assert r == expected_raster

    # validate we can construct the original raster from the chunks
    chunks = [r for r, _ in raster.chunks(pad=True)]
    merged_raster = join(chunks)
    assert merged_raster.crs == raster.crs
    assert merged_raster.affine.almost_equals(raster.affine, precision=1e-3)
    # raster created has bigger shape cause of the edges
    assert np.array_equal(merged_raster.image[:, :raster.height, :raster.width], raster.image)
    # validating the rest is masked
    assert np.array_equal(merged_raster.image.mask[:, raster.height:, raster.width:].all(), True)


@pytest.mark.parametrize("raster", rasters_for_testing_chunks())
def test_chunks_for_full_raster(raster):
    chunks = [r for r, _ in raster.chunks(pad=True)]
    merged_raster = join(chunks)
    joined_raster_from_256_mult_size = join([r for r, _ in merged_raster.chunks()])
    assert joined_raster_from_256_mult_size.shape == merged_raster.shape


@pytest.mark.parametrize("raster", rasters_for_testing_chunks())
def test_chunks_without_pad(raster):
    shape = (256, 256)

    # validate we can construct the original raster from the chunks
    chunks = [r for r, _ in raster.chunks()]
    merged_raster = join(chunks)
    assert merged_raster.crs == raster.crs
    assert merged_raster.affine.almost_equals(raster.affine, precision=1e-3)
    assert np.array_equal(merged_raster.image, raster.image)


def test_not_loades():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    assert raster.not_loaded()
    raster.image
    assert not raster.not_loaded()


def test_blockshapes_for_in_memory_raster():
    raster = GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['r'])
    assert len(raster.blockshapes) == raster.num_bands
    assert raster.blockshapes == [(raster.height, raster.width)]


def test_blockshapes_for_file_raster():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    assert len(raster.blockshapes) == raster.num_bands
    assert raster.blockshapes == [(27, 100), (27, 100), (27, 100)]


def test_from_assets_to_assets():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    assert raster == GeoRaster2.from_assets(raster.to_assets())


def test_copy_raster_without_crs(recwarn):
    raster = GeoRaster2.open("tests/data/raster/no_georef.png")
    raster_copy = raster.copy_with()
    assert raster_copy.crs == raster.crs


def test_virtual_filesystem_raster():
    raster_copy = some_raster.copy()
    in_memory_raster = raster_copy._as_in_memory_geotiff()
    assert len(raster_copy._opened_files) > 0
    virtual_path = in_memory_raster.source_file
    del raster_copy
    in_memory_raster2 = GeoRaster2.open(virtual_path)
    with pytest.raises(GeoRaster2IOError):
        _ = in_memory_raster2.shape
