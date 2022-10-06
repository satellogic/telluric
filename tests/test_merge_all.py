import unittest
import pytest
import copy

import numpy as np
from numpy.testing import assert_array_equal
from affine import Affine
from shapely.geometry import Polygon
from rasterio.crs import CRS

from telluric import FeatureCollection, GeoFeature
from telluric.constants import WEB_MERCATOR_CRS, WGS84_CRS
from telluric.vectors import GeoVector
from telluric.georaster import GeoRaster2, MergeStrategy, PixelStrategy, merge_all, merge_two

from common_for_tests import make_test_raster


def black_and_white_raster(band_names=[], height=10, width=10, dtype=np.uint16,
                           crs=WEB_MERCATOR_CRS, affine=None):
    if affine is None:
        eps = 1e-100
        affine = Affine.translation(10, 12) * Affine.scale(1, -1)
    bands_num = len(band_names)
    shape = [bands_num, height, width]
    array = np.zeros(shape, dtype=dtype)
    mask = np.full(shape, False, dtype=bool)
    val = 0
    for i in range(height):
        for j in range(width):
            for z in range(bands_num):
                array[z, i, j] = val
                val = 1 - val

    image = np.ma.array(data=array, mask=mask)
    raster = GeoRaster2(image=image, affine=affine, crs=crs, band_names=band_names)
    return raster


def test_merge_single_band_single_raster_returns_itself_for_all_strategies():
    for ms in MergeStrategy:
        raster = make_test_raster(88, [1])
        raster2 = merge_all([raster], roi=raster.footprint(), merge_strategy=ms)
        assert (raster2 == raster)


def test_merge_multi_band_single_raster_returns_itself_for_all_strategies():
    for ms in MergeStrategy:
        raster = black_and_white_raster([1, 2, 3])
        raster2 = merge_all([raster], roi=raster.footprint(), merge_strategy=ms)
        assert (raster2 == raster)


def test_merge_multi_band_multi_raster_returns_itself():
    rasters = [black_and_white_raster([1, 2, 3]) for i in range(10)]
    raster = black_and_white_raster([1, 2, 3])
    raster2 = merge_all(rasters, roi=raster.footprint())
    assert (raster2 == black_and_white_raster([1, 2, 3]))


def test_merge_multi_band_multi_raster_smaller_roi_returns_itself():
    rasters = [black_and_white_raster([1, 2, 3])]
    raster = black_and_white_raster([1, 2, 3], height=7, width=6)
    raster2 = merge_all(rasters, roi=raster.footprint())
    assert (raster2 == raster)


def get_rasters():
    rasters = [black_and_white_raster([1, 2, 3], height=100, width=100),
               black_and_white_raster([1, 2, 3], height=70, width=60),
               black_and_white_raster([1, 2, 3], height=130, width=60),
               black_and_white_raster([1, 2, 3], height=70, width=160)]
    return copy.deepcopy(rasters)


def test_merge_multi_band_multi_size_raster_0():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[0].footprint())
    assert (raster2 == rasters[0])


def test_merge_multi_band_multi_size_raster_1():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[1].footprint())
    assert (raster2 == rasters[1])


def test_merge_multi_band_multi_size_raster_2():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[2].footprint())
    assert (raster2 == rasters[2])


def test_merge_multi_band_multi_size_raster_3():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[3].footprint())
    assert (raster2 == rasters[3])


def test_empty_raster_from_roi_5_bands():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 2, 4, 5, 6], affine=affine, height=301, width=402)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert (affine.almost_equals(empty.affine))
    assert (raster.crs == empty.crs)
    assert (raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_wide():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 2], affine=affine, height=3, width=1402)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert (affine.almost_equals(empty.affine))
    assert (raster.crs == empty.crs)
    assert (raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_3_bands_high():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 3, 2], affine=affine, height=1301, width=4)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert (affine.almost_equals(empty.affine))
    assert (raster.crs == empty.crs)
    assert (raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_small():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1], affine=affine, height=31, width=42)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert (affine.almost_equals(empty.affine))
    assert (raster.crs == empty.crs)


@pytest.mark.parametrize("main_r", get_rasters())
@pytest.mark.parametrize("cropping_r", get_rasters())
def test_crop_for_merging(main_r, cropping_r):
    rr = main_r.crop(cropping_r.footprint(), resolution=cropping_r.resolution())
    assert (rr.height == min(main_r.height, cropping_r.height))
    assert (rr.width == min(main_r.width, cropping_r.width))
    assert (rr.num_bands == cropping_r.num_bands)
    assert (rr.affine.almost_equals(cropping_r.affine))


def test_pixel_crop():
    rr = black_and_white_raster([1, 2, 3], height=100, width=100)
    out = rr.pixel_crop((0, 0, 100, 100))
    assert (rr == out)
    out = rr.pixel_crop((0, 0, 10, 10), 10, 10, 1)
    assert (out.shape == (3, 10, 10))
    out = rr.pixel_crop((0, 0, 100, 100), 100, 100, 1)
    assert (rr == out)
    out = rr.pixel_crop((0, 0, 50, 50), 100, 100, 1)
    assert (out.shape == (3, 100, 100))


def test_patch_affine():
    eps = 1e-100
    assert (GeoRaster2._patch_affine(Affine.identity()) == Affine.translation(eps, eps))
    assert (GeoRaster2._patch_affine(Affine.translation(2 * eps, 3 * eps)) ==
           Affine.translation(2 * eps, 3 * eps))
    assert (GeoRaster2._patch_affine(Affine.translation(2, 3)) == Affine.translation(2, 3))
    assert (GeoRaster2._patch_affine(Affine.scale(1.0, -1)) ==
           Affine.translation(eps, -eps) * Affine.scale(1, -1))
    assert (GeoRaster2._patch_affine(Affine.scale(-1, 1)) ==
           Affine.translation(-eps, eps) * Affine.scale(-1, 1))
    assert (GeoRaster2._patch_affine(Affine.scale(-1, -1)) ==
           Affine.translation(-eps, -eps) * Affine.scale(-1, -1))
    assert (GeoRaster2._patch_affine(Affine.scale(1.1, -1)) == Affine.scale(1.1, -1))
    assert (GeoRaster2._patch_affine(Affine.scale(1, -1.1)) == Affine.scale(1, -1.1))


def test_rasters_covering_different_overlapping_areas_on_x():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=20, affine=affine_a)
    affine_b = Affine.translation(10, 2) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=20, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-8, xmax=30, ymax=2, crs=WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert (merged.affine.almost_equals(affine_a))
    assert (not merged.image.mask.all())
    assert ((merged.image.data[0, 0:10, 0:20] == 1).all())
    assert ((merged.image.data[0, 0:10, 21:30] == 2).all())


def test_rasters_covering_different_overlapping_areas_on_y():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=20, width=20, affine=affine_a)
    affine_b = Affine.translation(1, -9) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=20, width=20, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-29, xmax=21, ymax=2, crs=WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert (merged.affine.almost_equals(affine_a))
    assert (not merged.image.mask.all())
    assert ((merged.image.data[0, 0:20, 0:20] == 1).all())
    assert ((merged.image.data[0, 21:30, 0:20] == 2).all())


def test_rasters_covering_different_areas_with_gap_on_x():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=10, affine=affine_a)
    affine_b = Affine.translation(21, 2) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=10, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-8, xmax=30, ymax=2, crs=WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert (merged.affine.almost_equals(affine_a))
    assert (not merged.image.mask[0, 0:10, 0:10].all())
    assert (merged.image.mask[0, 0:10, 10:20].all())
    assert (not merged.image.mask[0, 0:10, 20:30].all())
    assert ((merged.image.data[0, 0:10, 0:10] == 1).all())
    assert ((merged.image.data[0, 0:10, 11:20] == 0).all())
    assert ((merged.image.data[0, 0:10, 21:30] == 2).all())


def test_rasters_covering_different_areas_with_gap_on_y():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=10, affine=affine_a)
    affine_b = Affine.translation(1, -19) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=10, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-29, xmax=11, ymax=2, crs=WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert (merged.affine.almost_equals(affine_a))
    assert (not merged.image.mask[0, 0:10, 0:10].all())
    assert (merged.image.mask[0, 11:20, 0:10].all())
    assert (not merged.image.mask[0, 21:30, 0:10].all())
    assert ((merged.image.data[0, 0:10, 0:10] == 1).all())
    assert ((merged.image.data[0, 11:20, 0:10] == 0).all())
    assert ((merged.image.data[0, 21:30, 0:10] == 2).all())


@unittest.skip("for manual testing of rasterio bug")
def test_rasterio_bug():
    import numpy as np
    import rasterio
    from affine import Affine

    eps = 1e-100
    data = np.full([3, 10, 11], 88, dtype=np.float32)
    dest_data = np.empty([3, 10, 11], dtype=np.float32)
    crs = rasterio.crs.CRS({'init': 'epsg:3857'})
    src_affine = Affine.scale(1, -1)
    src_affine_good = src_affine * Affine.translation(eps, eps)
    dst_affine = Affine.scale(0.5, -0.5)

    rasterio.warp.reproject(data, dest_data, src_transform=src_affine_good,
                            dst_transform=dst_affine, src_crs=crs, dst_crs=crs)

    src_affine_bad = Affine.translation(0, 0) * Affine.scale(1, -1)

    rasterio.warp.reproject(data, dest_data, src_transform=src_affine_bad,
                            dst_transform=dst_affine, src_crs=crs, dst_crs=crs)


def test_merge_raise_on_non_overlapping_rasters():
    affine1 = Affine.translation(10, 12) * Affine.scale(1, -1)
    affine2 = Affine.translation(100, 120) * Affine.scale(1, -1)
    raster1 = make_test_raster(affine=affine1)
    raster2 = make_test_raster(affine=affine2)
    with pytest.raises(ValueError) as ex:
        merge_two(raster1, raster2)

    assert "rasters do not intersect" in ex.exconly()


def test_merge_to_firs_on_non_overlapping_rasters_returns_first_raster():
    affine1 = Affine.translation(10, 12) * Affine.scale(1, -1)
    affine2 = Affine.translation(100, 120) * Affine.scale(1, -1)
    raster1 = make_test_raster(affine=affine1)
    raster2 = make_test_raster(affine=affine2)
    merged = merge_two(raster1, raster2, silent=True)
    assert merged == raster1


def test_merge_all_on_non_overlapping_rasters_returns_first_raster():
    affine1 = Affine.translation(10, 12) * Affine.scale(1, -1)
    affine2 = Affine.translation(100, 120) * Affine.scale(1, -1)
    raster1 = make_test_raster(value=1, band_names=['blue'], affine=affine1, height=30, width=40)
    raster2 = make_test_raster(value=2, band_names=['blue'], affine=affine2, height=30, width=40)
    merged = merge_all([raster1, raster2], raster1.footprint())
    assert merged == raster1


def test_merge_does_not_uncover_masked_pixels():
    # See https://github.com/satellogic/telluric/issues/65
    affine = Affine.translation(0, 2) * Affine.scale(1, -1)

    rs_a = GeoRaster2(
        image=np.ma.masked_array([
            [
                [100, 89],
                [100, 89]
            ],
            [
                [110, 99],
                [110, 99]
            ]
        ], [
            [
                [False, True],
                [False, True]
            ],
            [
                [False, True],
                [False, True]
            ]
        ], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red', 'green'],
    )

    rs_b = GeoRaster2(
        image=np.array([[
            [0, 210],
            [0, 210]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['green'],
    )

    expected_image = np.ma.masked_array([
        [
            [100, 89],
            [100, 89]
        ],
        [
            [110, 99],
            [110, 99]
        ]
        ], [
            [
                [False, True],
                [False, True]
            ],
            [
                [False, True],
                [False, True]
            ]
        ], dtype=np.uint8)

    result = merge_all([rs_a, rs_b], rs_a.footprint()).limit_to_bands(['red', 'green'])

    assert_array_equal(np.ma.filled(result.image, 0), np.ma.filled(expected_image, 0))
    assert_array_equal(result.image.mask, expected_image.mask)


def test_merge_all_non_overlapping_covers_all():
    # See https://github.com/satellogic/telluric/issues/65
    affine = Affine.translation(0, 2) * Affine.scale(1, -1)

    rs1 = GeoRaster2(
        image=np.array([[
            [100, 0],
            [100, 0]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red'],
        nodata=0,
    )

    rs2 = GeoRaster2(
        image=np.array([[
            [110, 0],
            [110, 0]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['green'],
        nodata=0,
    )

    rs3 = GeoRaster2(
        image=np.array([[
            [0, 200],
            [0, 200]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red'],
        nodata=0,
    )

    rs4 = GeoRaster2(
        image=np.array([[
            [0, 210],
            [0, 210]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['green'],
        nodata=0,
    )

    expected_image = np.ma.masked_array([
        [
            [100, 200],
            [100, 200]
        ],
        [
            [110, 210],
            [110, 210]
        ]
    ], False)

    result = merge_all([rs1, rs2, rs3, rs4], rs1.footprint()).limit_to_bands(['red', 'green'])

    assert_array_equal(result.image.data, expected_image.data)
    assert_array_equal(result.image.mask, expected_image.mask)


def test_merge_all_non_overlapping_has_correct_metadata():
    # See https://github.com/satellogic/telluric/issues/65
    affine = Affine.translation(0, 2) * Affine.scale(1, -1)

    rs1 = GeoRaster2(
        image=np.array([[
            [100, 0],
            [100, 0]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red'],
        nodata=0,
    )

    rs2 = GeoRaster2(
        image=np.array([[
            [110, 0],
            [110, 0]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['green'],
        nodata=0,
    )

    rs3 = GeoRaster2(
        image=np.array([[
            [0, 200],
            [0, 200]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red'],
        nodata=0,
    )

    rs4 = GeoRaster2(
        image=np.array([[
            [0, 210],
            [0, 210]
        ]], dtype=np.uint8),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['green'],
        nodata=0,
    )

    expected_metadata = GeoRaster2(
        image=np.ma.masked_array([
            [
                [0, 2],
                [0, 2],
            ],
            [
                [1, 3],
                [1, 3],
            ]
        ], np.ma.nomask),
        affine=affine,
        crs=WGS84_CRS,
        band_names=['red', 'green']
    )

    metadata = merge_all([rs1, rs2, rs3, rs4], rs1.footprint(), pixel_strategy=PixelStrategy.INDEX)

    assert metadata == expected_metadata


@pytest.mark.parametrize("crop", [True, False])
def test_merge_all_different_crs(crop, recwarn):
    roi = GeoVector(Polygon.from_bounds(-6321833, -3092272, -6319273, -3089712), WEB_MERCATOR_CRS)
    affine = Affine.translation(-57, -26) * Affine.scale(0.00083, -0.00083)
    expected_resolution = 10
    expected_crs = WEB_MERCATOR_CRS

    # from memory
    raster_0 = make_test_raster(1, [1], height=1200, width=1200, affine=affine, crs=WGS84_CRS)
    result_0 = merge_all([raster_0], roi=roi, dest_resolution=expected_resolution, crs=expected_crs, crop=crop)
    assert (result_0.resolution() == expected_resolution)
    assert (result_0.crs == expected_crs)
    assert (result_0.footprint().envelope.almost_equals(roi.envelope, decimal=3))

    # from file
    path = "/vsimem/raster_for_test.tif"
    result_0.save(path)
    raster_1 = GeoRaster2.open(path)
    result_1 = merge_all([raster_1], roi=roi, dest_resolution=expected_resolution, crs=expected_crs, crop=crop)

    assert (result_1.resolution() == expected_resolution)
    assert (result_1.crs == expected_crs)
    assert (result_1.footprint().envelope.almost_equals(roi.envelope, decimal=3))
    assert (result_0 == result_1)

    # preserve the original resolution if dest_resolution is not provided
    raster_2 = make_test_raster(1, [1], height=1200, width=1200, affine=affine, crs=WGS84_CRS)
    result_2 = merge_all([raster_2], roi=roi, crs=expected_crs, crop=crop)
    assert pytest.approx(result_2.resolution()) == 97.9691


def test_raster_closer_than_resolution_to_roi():
    raster_close_to_roi = make_test_raster(
        1,
        [1],
        height=2255,
        width=6500,
        affine=Affine(1.000056241624503, -0.0001677700491717716, 251130.52371896777,
                      -0.00011325628093143738, -1.0000703876618153, 2703061.4308057753),
        crs=CRS.from_epsg(32613),
    )
    raster_intersecting_roi = make_test_raster(
        1,
        [1],
        height=3515,
        width=6497,
        affine=Affine(1.000063460933417, -2.935588943753421e-05, 250953.40276071787,
                      -3.26265458078499e-05, -1.000053742629815, 2703428.138070052),
        crs=CRS.from_epsg(32613),
    )
    roi = GeoVector.from_bounds(251726, 2696110, 256422, 2700806, CRS.from_epsg(32613))
    merge_all(
        [raster_close_to_roi, raster_intersecting_roi],
        roi=roi,
        dest_resolution=(1, 1),
        merge_strategy=MergeStrategy.INTERSECTION,
    )


def test_rasters_close_than_resolution_to_roi_2():
    one_affine = Affine(1, 0, 598847, 0, -1, 3471062)
    other_affine = Affine(0.9998121393135052104028,
                          -0.000563382202975665213,
                          596893.24732190347276628,
                          -0.000249934917683214408,
                          -1.000473374252140335016,
                          3466367.0648421039804816)
    one = make_test_raster(1, [1], height=4696, width=4696, affine=one_affine, crs=CRS.from_epsg(32641))
    other = make_test_raster(2, [1], height=2616, width=5402, affine=other_affine, crs=CRS.from_epsg(32641))
    roi = GeoVector.from_bounds(598847.0000000002, 3466365.999999999, 603542.9999999995, 3471062, crs=one.crs)
    merged = merge_all([one, other], dest_resolution=(1, 1), roi=roi)
    assert merged == one
