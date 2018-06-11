import pytest
import os
from tempfile import TemporaryDirectory
from copy import copy, deepcopy

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from PIL import Image
from shapely.geometry import Point, Polygon

from telluric.constants import WGS84_CRS, WEB_MERCATOR_CRS
from telluric.georaster import GeoRaster2, GeoRaster2Error, GeoRaster2Warning
from telluric.vectors import GeoVector


some_array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
some_mask = np.array([[False, False, False], [False, False, True]], dtype=np.bool)
some_image_2d = np.ma.array(some_array, mask=some_mask)
some_image_2d_alt = np.ma.array(np.array([[0, 1, 2], [3, 4, 99]], dtype=np.uint8), mask=some_mask)
some_image_3d = np.ma.array(some_array[np.newaxis, :, :], mask=some_mask[np.newaxis, :, :])
some_image_3d_multiband = np.ma.array(
    np.array([some_array, some_array, some_array]), mask=np.array([some_mask, some_mask, some_mask]))
raster_origin = Point(2, 3)
some_affine = Affine.translation(raster_origin.x, raster_origin.y)
some_crs = {'init': 'epsg:32620'}
some_raster = GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['r'])
some_raster_alt = GeoRaster2(some_image_2d_alt, affine=some_affine, crs=some_crs, band_names=['r'])
some_raster_multiband = GeoRaster2(
    some_image_3d_multiband, band_names=['r', 'g', 'b'], affine=some_affine, crs=some_crs)
default_factors = [2, 4, 8, 16]


def make_test_raster(value=0, band_names=[], height=3, width=4, dtype=np.uint16,
                     crs=WEB_MERCATOR_CRS, affine=Affine.identity()):
    shape = [len(band_names), height, width]
    array = np.full(shape, value, dtype=dtype)
    mask = np.full(shape, False, dtype=np.bool)
    image = np.ma.array(data=array, mask=mask)
    raster = GeoRaster2(image=image, affine=affine, crs=crs, band_names=band_names)
    return raster


def test_construction():
    # test image - different formats yield identical rasters:
    raster_masked_2d = GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs)
    raster_masked_3d = GeoRaster2(some_image_3d, affine=some_affine, crs=some_crs)
    raster_masked_array = GeoRaster2(some_array, nodata=5, affine=some_affine, crs=some_crs)
    assert raster_masked_2d == raster_masked_3d
    assert raster_masked_2d == raster_masked_array

    assert np.array_equal(raster_masked_2d.image, some_image_3d)
    assert raster_masked_2d.affine == some_affine
    assert raster_masked_2d.crs == some_crs
    assert raster_masked_2d.dtype == some_image_2d.dtype

    # test bandnames:
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names='gray').band_names == ['gray']
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['gray']).band_names == ['gray']
    assert GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs).band_names == [0]
    with pytest.raises(GeoRaster2Error):
        GeoRaster2(some_image_2d, affine=some_affine, crs=some_crs, band_names=['gray', 'red'])


def test_eq():
    """ test ._eq_ """
    assert some_raster == some_raster
    assert some_raster != some_raster.copy_with(image=some_raster.image + 1)
    assert some_raster != some_raster.copy_with(affine=Affine.translation(42, 42))
    assert some_raster != some_raster.copy_with(crs={'init': 'epsg:32621'})


def test_eq_ignores_masked_values():
    assert some_raster == some_raster_alt


def test_read_write():
    for extension in ['tif', 'png']:
        with TemporaryDirectory() as folder:
            path = os.path.join(folder, 'test.%s' % extension)
            some_raster_multiband.save(path, factors=default_factors)
            read = GeoRaster2.open(path)
            assert read == some_raster_multiband


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


def test_tags():
    with TemporaryDirectory() as folder:
        path = os.path.join(folder, 'test.tif')
        some_raster_multiband.save(path, tags={'foo': 'bar'}, factors=default_factors)

        assert GeoRaster2.tags(path) == {'AREA_OR_POINT': 'Area', 'foo': 'bar',
                                         'telluric_band_names': '["r", "g", "b"]'}  # namespace=default
        assert GeoRaster2.tags(path, 'IMAGE_STRUCTURE') == {'COMPRESSION': 'LZW', 'INTERLEAVE': 'PIXEL'}


def test_copy():
    """ Tests .__copy__() and .__deepcopy__() """
    a_raster = some_raster.deepcopy_with()
    deep_copy = deepcopy(a_raster)
    # shallow_copy = copy(a_raster)
    a_raster.image.data[0, 0, 0] += 1
    # assert shallow_copy.image[0, 0, 0] == a_raster.image[0, 0, 0]
    assert deep_copy.image[0, 0, 0] == a_raster.image[0, 0, 0] - 1


def test_copy_with():
    new_affine = Affine.translation(42, 42)
    assert some_raster.copy_with(affine=new_affine).affine == new_affine


def test_resize():
    resampling_modes = [m for m in dir(Resampling) if not callable(getattr(Resampling, m)) and not m.startswith("__")]
    for resampling_name in resampling_modes:
        resampling = Resampling[resampling_name]
        print('\nresampling name:', resampling_name)
        if resampling_name == 'gauss':
            print('\nskipping', resampling_name)
            continue
        if resampling_name not in ['bilinear', 'cubic', 'cubic_spline', 'lanczos', 'gauss']:
            resized_raster = some_raster.resize(2, resampling=resampling).resize(.5, resampling=resampling)
            assert (some_raster.image == resized_raster.image).all()
            assert some_raster == resized_raster
            # continue
        if resampling_name not in ['cubic_spline']:
            assert some_raster.resize(ratio=1, resampling=resampling) == some_raster

        assert some_raster.resize(ratio=2, resampling=resampling).width == 2 * some_raster.width
        assert some_raster.resize(ratio=2, resampling=resampling).shape == (1, 4, 6)
        assert some_raster.resize(dest_height=42, resampling=resampling).height == 42
        assert some_raster.resize(dest_width=42, resampling=resampling).width == 42
        assert some_raster.resize(dest_width=42, dest_height=42, resampling=resampling).width == 42
        assert some_raster.resize(dest_width=42, dest_height=42, resampling=resampling).height == 42
        assert some_raster.resize(dest_resolution=42, resampling=resampling).resolution() == 42

    with pytest.raises(GeoRaster2Error):
        some_raster.resize(ratio=1, dest_width=2)
    with pytest.raises(GeoRaster2Error):
        some_raster.resize(ratio_x=2)


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


def test_to_png():
    for raster in [some_raster, some_raster_multiband]:
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
    assert str(w.message) == "Limiting two bands raster to use the first band to generate png"

    w = recwarn.pop(GeoRaster2Warning)
    assert str(w.message) == "downscaling dtype to 'uint8' to convert to png"

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
    assert back_in_image.almost_equals(pt)


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
        assert some_raster.image_corner(corner).almost_equals(expected_image_corners[corner])


def test_center():
    ul = some_raster.corner('ul').get_shape(some_raster.crs)
    br = some_raster.corner('br').get_shape(some_raster.crs)
    expected_center = Point((ul.x + br.x) / 2, (ul.y + br.y) / 2)
    expected_center_vector = GeoVector(expected_center, some_raster.crs)

    assert expected_center_vector.almost_equals(some_raster.center())


def test_bounds():
    expected = Polygon([[0, 0],
                        [some_raster.width, 0],
                        [some_raster.width, some_raster.height],
                        [0, some_raster.height]])

    assert some_raster.bounds().almost_equals(expected)


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

    # raster masked "inside shape" should be inverse to raster masked "outside shape":
    assert raster.mask(vector, mask_shape_nodata=True) == ~raster.mask(vector)


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


def test_astype_uint8_to_float32_conversion():
    raster_uint8 = make_test_raster(42, band_names=[1, 2], dtype=np.uint8)
    with pytest.raises(GeoRaster2Error) as err:
        raster_uint8.astype(np.float32)
    expected_error_message = 'astype to non integer type is not supported - requested dtype: %s' % np.float32
    print(err.exconly())
    assert expected_error_message in err.exconly()


def test_astype_float32_to_uint8_conversion():
    with pytest.raises(GeoRaster2Error) as err:
        raster_float32 = make_test_raster(1.42, band_names=[1, 2], dtype=np.float32)
        raster_float32.astype(np.uint8)
    expected_error_message = \
        'astype from non integer type is not supported - raster dtype: %s' % raster_float32.image.dtype
    assert expected_error_message in err.exconly()


def test_png_thumbnail_has_expected_properties():
    raster = GeoRaster2.open("tests/data/raster/rgb.tif")
    expected_thumbnail = raster.resize(dest_width=512, resampling=Resampling.nearest)
    result_thumbnail = GeoRaster2.from_bytes(
        raster._repr_png_(),
        affine=expected_thumbnail.affine, crs=expected_thumbnail.crs, band_names=expected_thumbnail.band_names
    )

    assert result_thumbnail == expected_thumbnail
