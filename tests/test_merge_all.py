import unittest
import pytest
import copy
import numpy as np
from affine import Affine
from telluric import constants, GeoVector
from telluric.georaster import GeoRaster2, MergeStrategy, merge_all, merge, merge_to_first
from common_for_tests import make_test_raster

def black_and_white_raster(band_names=[], height=10, width=10, dtype=np.uint16,
                           crs=constants.WEB_MERCATOR_CRS, affine=None):
    if affine is None:
        eps = 1e-100
        affine = Affine.translation(10, 12) * Affine.scale(1, -1)
    bands_num = len(band_names)
    shape = [bands_num, height, width]
    array = np.zeros(shape, dtype=dtype)
    mask = np.full(shape, False, dtype=np.bool)
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
        assert(raster2 == raster)


def test_merge_multi_band_single_raster_returns_itself_for_all_strategies():
    for ms in MergeStrategy:
        raster = black_and_white_raster([1, 2, 3])
        raster2 = merge_all([raster], roi=raster.footprint(), merge_strategy=ms)
        assert(raster2 == raster)


def test_merge_multi_band_multi_raster_returns_itself():
    rasters = [black_and_white_raster([1, 2, 3]) for i in range(10)]
    raster = black_and_white_raster([1, 2, 3])
    raster2 = merge_all(rasters, roi=raster.footprint())
    assert(raster2 == black_and_white_raster([1, 2, 3]))


def test_merge_multi_band_multi_raster_smaller_roi_returns_itself():
    rasters = [black_and_white_raster([1, 2, 3])]
    raster = black_and_white_raster([1, 2, 3], height=7, width=6)
    raster2 = merge_all(rasters, roi=raster.footprint())
    assert(raster2 == raster)


def get_rasters():
    rasters = [black_and_white_raster([1, 2, 3], height=1000, width=1000),
               black_and_white_raster([1, 2, 3], height=700, width=600),
               black_and_white_raster([1, 2, 3], height=1300, width=600),
               black_and_white_raster([1, 2, 3], height=700, width=1600)]
    return copy.deepcopy(rasters)


def test_merge_multi_band_multi_size_raster_0():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[0].footprint())
    assert(raster2 == rasters[0])


def test_merge_multi_band_multi_size_raster_1():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[1].footprint())
    assert(raster2 == rasters[1])


def test_merge_multi_band_multi_size_raster_2():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[2].footprint())
    assert(raster2 == rasters[2])


def test_merge_multi_band_multi_size_raster_3():
    rasters = get_rasters()
    raster2 = merge_all(rasters, roi=rasters[3].footprint())
    assert(raster2 == rasters[3])


def test_empty_raster_from_roi_5_bands():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 2, 4, 5, 6], affine=affine, height=301, width=402)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert(affine.almost_equals(empty.affine))
    assert(raster.crs == empty.crs)
    assert(raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_wide():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 2], affine=affine, height=3, width=1402)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert(affine.almost_equals(empty.affine))
    assert(raster.crs == empty.crs)
    assert(raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_3_bands_high():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1, 3, 2], affine=affine, height=1301, width=4)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert(affine.almost_equals(empty.affine))
    assert(raster.crs == empty.crs)
    assert(raster.shape == empty.shape)


def test_empty_raster_from_roi_affine_small():
    affine = Affine.translation(10, 12) * Affine.scale(2, -2)
    raster = make_test_raster(88, [1], affine=affine, height=31, width=42)
    empty = GeoRaster2.empty_from_roi(band_names=raster.band_names, roi=raster.footprint(), resolution=2)
    assert(affine.almost_equals(empty.affine))
    assert(raster.crs == empty.crs)


def test_crop_for_merging():
    rasters = get_rasters()
    for i in range(len(rasters)):
        for j in range(len(rasters)):
            main_r = rasters[i]
            cropping_r = rasters[j]
            rr = main_r.crop(cropping_r.footprint(), resolution=cropping_r.resolution())
            assert(rr.height == min(main_r.height, cropping_r.height))
            assert(rr.width == min(main_r.width, cropping_r.width))
            assert(rr.num_bands == cropping_r.num_bands)
            assert(rr.affine.almost_equals(cropping_r.affine))


def test_pixel_crop():
    rr = black_and_white_raster([1, 2, 3], height=1000, width=1000)
    out = rr.pixel_crop((0, 0, 1000, 1000))
    assert(rr == out)
    out = rr.pixel_crop((0, 0, 100, 100), 100, 100, 1)
    assert(out.shape == (3, 100, 100))
    out = rr.pixel_crop((0, 0, 1000, 1000), 1000, 1000, 1)
    assert(rr == out)
    out = rr.pixel_crop((0, 0, 500, 500), 1000, 1000, 1)
    assert(out.shape == (3, 1000, 1000))


def test_patch_affine():
    eps = 1e-100
    assert(GeoRaster2._patch_affine(Affine.identity()) == Affine.translation(eps, eps))
    assert(GeoRaster2._patch_affine(Affine.translation(2 * eps, 3 * eps)) ==
           Affine.translation(2 * eps, 3 * eps))
    assert(GeoRaster2._patch_affine(Affine.translation(2, 3)) == Affine.translation(2, 3))
    assert(GeoRaster2._patch_affine(Affine.scale(1.0, -1)) ==
           Affine.translation(eps, -eps) * Affine.scale(1, -1))
    assert(GeoRaster2._patch_affine(Affine.scale(-1, 1)) ==
           Affine.translation(-eps, eps) * Affine.scale(-1, 1))
    assert(GeoRaster2._patch_affine(Affine.scale(-1, -1)) ==
           Affine.translation(-eps, -eps) * Affine.scale(-1, -1))
    assert(GeoRaster2._patch_affine(Affine.scale(1.1, -1)) == Affine.scale(1.1, -1))
    assert(GeoRaster2._patch_affine(Affine.scale(1, -1.1)) == Affine.scale(1, -1.1))


def test_rasters_covering_different_overlapping_areas_on_x():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=20, affine=affine_a)
    affine_b = Affine.translation(10, 2) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=20, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-8, xmax=30, ymax=2, crs=constants.WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert(merged.affine.almost_equals(affine_a))
    assert(not merged.image.mask.all())
    assert((merged.image.data[0, 0:10, 0:20] == 1).all())
    assert((merged.image.data[0, 0:10, 21:30] == 2).all())


def test_rasters_covering_different_overlapping_areas_on_y():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=20, width=20, affine=affine_a)
    affine_b = Affine.translation(1, -9) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=20, width=20, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-29, xmax=21, ymax=2, crs=constants.WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert(merged.affine.almost_equals(affine_a))
    assert(not merged.image.mask.all())
    assert((merged.image.data[0, 0:20, 0:20] == 1).all())
    assert((merged.image.data[0, 21:30, 0:20] == 2).all())


def test_rasters_covering_different_areas_with_gap_on_x():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=10, affine=affine_a)
    affine_b = Affine.translation(21, 2) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=10, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-8, xmax=30, ymax=2, crs=constants.WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert(merged.affine.almost_equals(affine_a))
    assert(not merged.image.mask[0, 0:10, 0:10].all())
    assert(merged.image.mask[0, 0:10, 10:20].all())
    assert(not merged.image.mask[0, 0:10, 20:30].all())
    assert((merged.image.data[0, 0:10, 0:10] == 1).all())
    assert((merged.image.data[0, 0:10, 11:20] == 0).all())
    assert((merged.image.data[0, 0:10, 21:30] == 2).all())


def test_rasters_covering_different_areas_with_gap_on_y():
    affine_a = Affine.translation(1, 2) * Affine.scale(1, -1)
    raster_a = make_test_raster(1, [1], height=10, width=10, affine=affine_a)
    affine_b = Affine.translation(1, -19) * Affine.scale(1, -1)
    raster_b = make_test_raster(2, [1], height=10, width=10, affine=affine_b)
    roi = GeoVector.from_bounds(xmin=1, ymin=-29, xmax=11, ymax=2, crs=constants.WEB_MERCATOR_CRS)
    rasters = [raster_a, raster_b]
    merged = merge_all(rasters, roi)
    assert(merged.affine.almost_equals(affine_a))
    assert(not merged.image.mask[0, 0:10, 0:10].all())
    assert(merged.image.mask[0, 11:20, 0:10].all())
    assert(not merged.image.mask[0, 21:30, 0:10].all())
    assert((merged.image.data[0, 0:10, 0:10] == 1).all())
    assert((merged.image.data[0, 11:20, 0:10] == 0).all())
    assert((merged.image.data[0, 21:30, 0:10] == 2).all())


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
        merge(raster1, raster2)

    assert "rasters do not intersect" in ex.exconly()


def test_merge_to_firs_on_non_overlapping_rasters_returns_first_raster():
    affine1 = Affine.translation(10, 12) * Affine.scale(1, -1)
    affine2 = Affine.translation(100, 120) * Affine.scale(1, -1)
    raster1 = make_test_raster(affine=affine1)
    raster2 = make_test_raster(affine=affine2)
    merged = merge_to_first(raster1, raster2)
    assert merged == raster1


def test_merge_all_on_non_overlapping_rasters_returns_first_raster():
    affine1 = Affine.translation(10, 12) * Affine.scale(1, -1)
    affine2 = Affine.translation(100, 120) * Affine.scale(1, -1)
    raster1 = make_test_raster(value=1, band_names=['blue'], affine=affine1, height=30, width=40)
    raster2 = make_test_raster(value=2, band_names=['blue'], affine=affine2, height=30, width=40)
    merged = merge_all([raster1, raster2], raster1.footprint())
    assert merged == raster1
