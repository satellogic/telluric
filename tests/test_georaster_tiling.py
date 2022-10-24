import os
import rasterio
import mercantile
import numpy as np

import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory

from affine import Affine

from unittest import TestCase
from unittest.mock import patch
from datetime import datetime
from shapely.geometry import Polygon
from packaging import version

from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.crs import CRS

from telluric import GeoRaster2, GeoVector
from telluric.constants import WEB_MERCATOR_CRS, WGS84_CRS
from telluric.georaster import MERCATOR_RESOLUTION_MAPPING, GeoRaster2Error, GeoRaster2IOError
from telluric.util.general import convert_resolution_from_meters_to_deg

import sys
import logging
import tempfile


log = logging.getLogger('rasterio._gdal')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


manualtest = pytest.mark.skipif("TEST_MANUAL" not in os.environ, reason="skip on auto testing")
window_data = pytest.mark.skip('pending decission of consistency in results between rasterio read and reproject')
framing = pytest.mark.skip('witing for framing and get_window with boundless false')

tiles = {
    10: (579, 394, 10),
    11: (1159, 789, 11),
    12: (2319, 1578, 12),
    14: (9277, 6312, 14),
    15: (18554, 12624, 15),
    17: (74216, 50496, 17),
    18: (148433, 100994, 18)
}


class GeoRaster2TilesTestGeneral(TestCase):
    """GeoRaster2 Tiles general tests."""

    def test_raise_exception_on_bad_file_path(self):
        vr = GeoRaster2.open('stam')
        with self.assertRaises(GeoRaster2IOError):
            vr.get_tile(1, 2, 3)

    def test_raise_exception_on_bad_raster_url(self):
        vr = GeoRaster2.open('http://stam')
        with self.assertRaises(GeoRaster2IOError):
            vr.get_tile(1, 2, 3)

    def test_raise_exception_on_bad_file_path_save_cog(self):
        vr = GeoRaster2.open('stam')
        with self.assertRaises(GeoRaster2IOError):
            vr.save_cloud_optimized('dest_file')

    def test_raise_exception_on_bad_raster_url_save_cog(self):
        vr = GeoRaster2.open('http://stam')
        with self.assertRaises(GeoRaster2IOError):
            vr.save_cloud_optimized('dest_file')


class BaseGeoRasterTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        path = "./tests/data/raster/raster_for_test.tif"
        cls.read_only_vgr = GeoRaster2.open(path)
        path = "./tests/data/raster/raster_wgs84.tif"
        cls.read_only_vgr_wgs84 = GeoRaster2.open(path)

    def read_only_virtual_geo_raster(self):
        return self.read_only_vgr

    def read_only_virtual_geo_raster_wgs84(self):
        return self.read_only_vgr_wgs84


class GeoRaster2TestGetTile(BaseGeoRasterTestCase):
    """GeoRaster2 get tile tests."""

    def test_geo_bounding_tile(self):
        gr = self.read_only_virtual_geo_raster()
        gv = gr.footprint().reproject(WGS84_CRS)
        bounding_tile = mercantile.bounding_tile(*gv.get_shape(gv.crs).bounds)
        self.assertEqual(bounding_tile, (37108, 25248, 16))

    @patch.object(GeoRaster2, 'crop')
    def test_fails_with_empty_raster_for_tile_out_of_raster_area(self, mock__crop):
        for raster in [self.read_only_virtual_geo_raster(), self.read_only_virtual_geo_raster_wgs84()]:
            r = raster.get_tile(16384, 16383, 15)
            self.assertTrue((r.image.data == 0).all())
            self.assertTrue((r.image.mask).all())
            self.assertEqual(r.image.shape, (3, 256, 256))
            self.assertEqual(r.crs, WEB_MERCATOR_CRS)
            mock__crop.assert_not_called()

    def test_get_all_raster_in_a_single_tile(self):
        for raster in [self.read_only_virtual_geo_raster(), self.read_only_virtual_geo_raster_wgs84()]:
            p = raster.footprint().reproject(WGS84_CRS).centroid
            r = raster.get_tile(*mercantile.tile(lng=p.x, lat=p.y, zoom=11))
            self.assertFalse((r.image.data == 0).all())
            self.assertFalse((r.image.mask).all())
            self.assertEqual(r.image.shape, (3, 256, 256))
            self.assertEqual(r.crs, WEB_MERCATOR_CRS)

    def test_get_tile_for_different_zoom_levels(self):
        for raster in [self.read_only_virtual_geo_raster(), self.read_only_virtual_geo_raster_wgs84()]:
            for zoom in tiles:
                r = raster.get_tile(*tiles[zoom])
                self.assertFalse((r.image.data == 0).all())
                self.assertFalse((r.image.mask).all())
                self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_tile_from_different_crs_tile_is_not_tilted(self):
        raster = self.read_only_virtual_geo_raster_wgs84()
        r = raster.get_tile(*tiles[18])
        self.assertEqual(1, len(np.unique(r.image.mask)))

    def test_get_tile_from_different_crs_tile_is_not_tilted_with_different_buffer(self):
        raster = self.read_only_virtual_geo_raster_wgs84()
        os.environ["TELLURIC_GET_TILE_BUFFER"] = "0"
        try:
            r = raster.get_tile(*tiles[18])
        except Exception:
            del os.environ["TELLURIC_GET_TILE_BUFFER"]
        self.assertEqual(2, len(np.unique(r.image.mask)))

    @pytest.mark.skipif(
        version.parse(rasterio.__version__) >= version.parse('1.3'),
        reason="rasterio >= 1.3 produces (3, 611, 611) shape image",
    )
    def test_get_entire_all_raster(self):
        vr = self.read_only_virtual_geo_raster()
        roi = GeoVector.from_xyz(37108, 25248, 16)
        r = vr.crop(roi)

        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.shape, (3, 612, 612))

    def test_fails_with_empty_raster_for_tile_out_of_raster_area_with_no_tile_size(self):
        vr = self.read_only_virtual_geo_raster()
        roi = GeoVector.from_xyz(16384, 16383, 15)
        r = vr.crop(roi)
        self.assertTrue((r.image.data == 0).all())
        self.assertTrue((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 1223, 1223))

    def test_get_window_of_full_resolution(self):
        vr = self.read_only_virtual_geo_raster()
        win = Window(0, 0, 300, 300)
        r = vr.get_window(win)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 300, 300))

    def test_get_window_resize_to_256(self):
        vr = self.read_only_virtual_geo_raster()
        win = Window(0, 0, 300, 300)
        r = vr.get_window(win, xsize=256, ysize=256)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_window_of_non_square_resize_to_256(self):
        vr = self.read_only_virtual_geo_raster()
        win = Window(0, 0, 300, 400)
        r = vr.get_window(win, xsize=256, ysize=256)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_window_of_non_square_keeps_size_proportions_for_give_xsize(self):
        vr = self.read_only_virtual_geo_raster()
        win = Window(0, 0, 300, 400)
        r = vr.get_window(win, xsize=150)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 200, 150))

    def test_get_window_of_non_square_keeps_size_proportions_for_give_ysize(self):
        vr = self.read_only_virtual_geo_raster()
        win = Window(0, 0, 300, 400)
        r = vr.get_window(win, ysize=200)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 200, 150))

    def test_get_window_width_height_correctness(self):
        # See https://publicgitlab.satellogic.com/telluric/telluric/issues/58
        vr = self.read_only_virtual_geo_raster()
        expected_height = 200
        win = Window(0, vr.height - expected_height, 1, expected_height)
        r = vr.get_window(win)
        self.assertEqual(r.image.shape, (3, expected_height, 1))


class GeoRasterCropTest(BaseGeoRasterTestCase):
    metric_affine = Affine(1, 0.0, 2653750, 0.0, -1, 4594461)

    def test_crop_in_memory_and_off_memory_without_resizing_are_the_same(self):
        coords = mercantile.xy_bounds(*tiles[18])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster2 = GeoRaster2.open(rf.name)
            off_memory_crop = raster2.crop(shape)
            # load the image data
            raster2.image
            in_memory_crop = raster2.crop(shape)
            self.assertEqual(off_memory_crop, in_memory_crop)

    @window_data
    def test_crop_and_get_tile_do_the_same(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster2 = GeoRaster2.open(rf.name)
            tile15 = raster2.get_tile(*tiles[15])
            # load the image data
            raster2.image
            cropped15 = raster2.crop(shape, MERCATOR_RESOLUTION_MAPPING[15])
            self.assertEqual(tile15, cropped15)

    @window_data
    def test_crop_and_get_tile_do_the_same_when_image_is_populated(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            tile15 = raster.get_tile(*tiles[15])
            raster._populate_from_rasterio_object(read_image=True)
            cropped_15 = raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[15])
            self.assertEqual(tile15, cropped_15)

    @window_data
    def test_crop_image_from_and_get_win_do_the_same_with_resize(self):
        bounds = (2, 3, 4, 5)
        win = rasterio.windows.Window(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        xsize = round((bounds[2] - bounds[0]) / 2)
        ysize = round((bounds[3] - bounds[1]) / 2)
        raster = self.read_only_virtual_geo_raster()

        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster.save('area.tif', tags={'AREA_OR_POINT': 'Area'})
            raster.save('point.tif', tags={'AREA_OR_POINT': 'Point'})
            saved_raster = GeoRaster2.open(rf.name)
            cropped_win = saved_raster.get_window(win, xsize=xsize, ysize=ysize)
            saved_raster_area = GeoRaster2.open('area.tif')
            cropped_win_area = saved_raster_area.get_window(win, xsize=xsize, ysize=ysize)
            saved_raster_point = GeoRaster2.open('point.tif')
            cropped_win_point = saved_raster_point.get_window(win, xsize=xsize, ysize=ysize)

        cropped_image = raster._crop(bounds, xsize=xsize, ysize=ysize)

        print('cropped_win_area pixels\n', cropped_win_area.image)
        print('cropped_win_point pixels\n', cropped_win_point.image)
        print('cropped_win pixels\n', cropped_win.image)
        print('cropped_image pixels\n', cropped_image.image)
        if (cropped_win_point == cropped_win_area):
            print('point == area')
        if (cropped_image == cropped_win_area):
            print('image == area')
        if (cropped_image == cropped_win_point):
            print('image == point')
        if (cropped_win == cropped_win_area):
            print('win == area')
        if (cropped_win == cropped_win_point):
            print('win == point')

        self.assertEqual(cropped_image, cropped_win)

    @framing
    def test_crop_and_get_tile_do_the_same_when_image_is_populated_first_high_zoom(self):
        coords = mercantile.xy_bounds(*tiles[17])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile17 = raster.get_tile(*tiles[17])
            cropped_17 = raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[17])
            self.assertEqual(tile17, cropped_17)

    @framing
    def test_crop_and_get_tile_do_the_same_when_image_is_populated_first_mid_zoom(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile15 = raster.get_tile(*tiles[15])
            cropped_15 = raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[15])
            self.assertEqual(tile15, cropped_15)

    @framing
    def test_crop_and_get_tile_do_the_same_when_image_is_populated_first_for_low_zoom(self):
        coords = mercantile.xy_bounds(*tiles[11])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile11 = raster.get_tile(*tiles[11])
            cropped_11 = raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[11])
            self.assertEqual(tile11, cropped_11)

    def test_crop_image_from_and_get_win_do_the_same_full_resolution(self):
        bounds = (20, 13, 40, 15)
        win = rasterio.windows.Window(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            saved_raster = GeoRaster2.open(rf.name)
            cropped_win = saved_raster.get_window(win)
        cropped_image = raster._crop(bounds)
        self.assertEqual(cropped_image, cropped_win)

    @patch.object(GeoRaster2, '_crop')
    def test_crop_use_crop_image_for_a_loaded_image(self, mock__crop):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[15])
        assert mock__crop.called_once

    @patch.object(GeoRaster2, 'get_window')
    def test_crop_use_get_window_for_a_not_loaded_image(self, mock_get_window):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[15])
            assert mock_get_window.called_once

    def test_crop_returns_full_resolution_as_default(self):
        coords = mercantile.xy_bounds(*tiles[17])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        _, win = raster._vector_to_raster_bounds(shape)
        cropped = raster.crop(shape)
        self.assertEqual(cropped.shape, (raster.num_bands, round(win.height), round(win.width)))
        self.assertEqual(cropped.affine[0], raster.affine[0])

    def test_memory_crop_returns_resized_resolution(self):
        coords = mercantile.xy_bounds(*tiles[18])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.read_only_virtual_geo_raster()
        cropped = raster.crop(shape, MERCATOR_RESOLUTION_MAPPING[18])
        self.assertEqual(cropped.shape, (raster.num_bands, 256, 256))
        self.assertAlmostEqual(cropped.affine[0], MERCATOR_RESOLUTION_MAPPING[18], 2)

    def test_geographic_crop(self):
        raster = self.read_only_virtual_geo_raster_wgs84()
        rhombus_on_image = Polygon([[0, 2], [1, 1], [2, 2], [1, 3]])  # in pixels
        rhombus_world = raster.to_world(rhombus_on_image)
        cropped = raster.crop(rhombus_world)
        r = raster[0:2, 1:3]
        assert cropped == r

    def test_geographic_crop_with_resize(self):
        coords = mercantile.xy_bounds(*tiles[17])
        raster = self.read_only_virtual_geo_raster_wgs84()
        vector = GeoVector(Polygon.from_bounds(*coords), crs=WEB_MERCATOR_CRS)
        x_ex_res, y_ex_res = convert_resolution_from_meters_to_deg(
            self.metric_affine[6], MERCATOR_RESOLUTION_MAPPING[17])
        cropped = raster.crop(vector, (x_ex_res, y_ex_res))
        self.assertAlmostEqual(cropped.affine[0], x_ex_res)
        self.assertAlmostEqual(abs(cropped.affine[4]), y_ex_res, 6)

    @pytest.mark.skipif(
        version.parse(rasterio.__version__) >= version.parse('1.3'),
        reason="rasterio >= 1.3 doesn't raise exception on given transformation",
    )
    def test_crop_raises_error_for_impossible_transformation(self):
        raster = self.read_only_virtual_geo_raster()
        vector = GeoVector(Polygon.from_bounds(-180, -90, 180, 90), crs=WGS84_CRS)
        with self.assertRaises(GeoRaster2Error):
            raster.crop(vector)

    def test_crop_of_rasters_with_opposite_affine_and_data_return_the_same(self):
        array = np.arange(0, 20).reshape(1, 4, 5)
        array2 = np.arange(19, -1, -1).reshape(1, 4, 5)
        array2.sort()

        image1 = np.ma.array(array, mask=False)
        image2 = np.ma.array(array2, mask=False)

        aff2 = Affine.translation(0, -8) * Affine.scale(2, 2)
        aff = Affine.scale(2, -2)

        r1 = GeoRaster2(image=image1, affine=aff, crs=WEB_MERCATOR_CRS)
        r2 = GeoRaster2(image=image2, affine=aff2, crs=WEB_MERCATOR_CRS)

        # r1 == r2  # doesn't work, see https://github.com/satellogic/telluric/issues/79
        roi = GeoVector(Polygon.from_bounds(0, 0, 3, -3), crs=WEB_MERCATOR_CRS)

        r1c = r1.crop(roi)
        r2c = r2.crop(roi)

        # r1c == r2c  # doesn't work, see https://github.com/satellogic/telluric/issues/79
        # currently this is the only way to test the result is the same
        assert np.all(np.flip(r1c.image, axis=1) == r2c.image)


class GeoRasterMaskedTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = TemporaryDirectory()
        path = os.path.join(cls.dir.name, 'test_masked_raster.tif')
        cls.masked_raster().save(path)
        cls.read_only_vgr = GeoRaster2.open(path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    @classmethod
    def masked_raster(cls):
        data = np.array([
            [0, 1, 1, 1],
            [0, 2, 0, 2],
            [0, 3, 3, 3],
        ], dtype=np.uint8)

        mask = np.array([
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ], dtype=bool)

        image = np.ma.array(
            np.repeat(data[np.newaxis, :, :], 3, 0),
            mask=np.repeat(mask[np.newaxis, :, :], 3, 0)
        )

        # Don't use exactly -1.0 for the affine for rasterio < 1.0a13, see
        # https://github.com/mapbox/rasterio/issues/1272
        affine = Affine.scale(1, -1.0001) * Affine.translation(0, -3)
        crs = WGS84_CRS

        return GeoRaster2(
            image, affine=affine, crs=crs,
        )

    def read_only_virtual_geo_raster(self):
        return self.read_only_vgr

    def test_get_smaller_window_respects_mask(self):
        window = Window(1, 0, 3, 3)
        raster = self.read_only_virtual_geo_raster()

        cropped = raster.get_window(window, masked=True)

        assert (~cropped.image.mask).all()

    def test_get_bigger_window_respects_mask(self):
        window = Window(1, 0, 4, 3)
        raster = self.read_only_virtual_geo_raster()

        cropped = raster.get_window(window, masked=True)

        assert cropped.image[:, :, -1].mask.all()  # This line of pixels is masked
        assert (~cropped.image[:, :, :-1].mask).all()  # The rest is not masked


def test_small_read_only_virtual_geo_raster_wgs84_crop():
    # See https://github.com/satellogic/telluric/issues/61
    roi = GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=2, crs=WGS84_CRS)
    resolution = 1.0  # deg / px

    raster = GeoRaster2.empty_from_roi(roi, resolution)

    assert raster.crop(roi) == raster.crop(roi, raster.resolution())


@manualtest
class GeoRaster2ManualTest(TestCase):
    """manual testing To be run manually only."""

    files = {
        'original': 'original2.tif',
        'cloudoptimized aligned': 'original2_aligned_cloudoptimized-2.tif',
        'mrf aligned': 'original2_aligned.mrf',
        'cloudoptimized': 'original2_cloudoptimized-2.tif',
        'mrf': 'original2.mrf',
        'not aligned cloudoptimized': 'not_aligned_cloudoptimized_2.tif',
        'not aligned mrf': 'not_aligned.mrf',
        'not aligned mrf split': 'not_aligned_split.mrf',
        'aligned mrf split': 'original2_aligned_split.mrf',
        'original mrf split': 'original2_split.mrf',
    }

    resamplings = {
        # 'avarage': Resampling.average,
        # 'nearest': Resampling.nearest,
        # 'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }

    def random_string(self):
        import hashlib
        now = '%s' % datetime.now()
        return hashlib.md5(now.encode('utf-8')).hexdigest()

    def run_test_on_real_rasters(self, zoom, resampling, local):
        results_arr = np.empty(shape=(len(self.files)), dtype=object)
        # with rasterio.Env(CPL_DEBUG=True, GDAL_CACHEMAX=0):
        # with rasterio.Env(CPL_DEBUG=False):
        print('*' * 80)
        print(zoom)
        print('*' * 80)
        print('#' * 80)
        print(resampling.name)
        print('#' * 80)
        for i, (file_type, file_url) in enumerate(self.files.items()):
            if local or 'split' in file_type:
                base_url = './notebooks/'
            else:
                base_url = 'https://ariel.blob.core.windows.net/rastersfortest/'
            file_url = base_url + file_url
            if local and 'mrf' not in file_type:
                new_file = file_url + self.random_string()
                os.system("cp %s %s" % (file_url, new_file))
            else:
                new_file = file_url

            print('file type: %s' % file_type)
            print('-' * 80)
            print('file_url: %s' % file_url)
            print('new_file: %s' % new_file)
            print('-' * 80)
            vr = GeoRaster2.open(new_file)
            start = datetime.now()
            rasterio_ops = {
                'CPL_DEBUG': True,
                'GDAL_DISABLE_READDIR_ON_OPEN': 'YES'
            }
            if 'mrf' not in file_type:
                rasterio_ops['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'
            with rasterio.Env(**rasterio_ops):
                vr.get_tile(*tiles[zoom], resampling=resampling)
            end = datetime.now()
            tt = (end - start).total_seconds() * 1000
            print("stars time : %s end time: %s total: %s ms" % (start, end, tt))
            results_arr[i] = "type: %s, zoom: %i, resampling: %s time: %s msec" % (file_type, zoom,
                                                                                   resampling.name, tt)
            if local and 'mrf' not in file_type:
                os.system("rm -f %s" % (new_file))

            print('=' * 80)
        print(results_arr)

    def test_zoom_remote_11_resampling_cubic(self):
        self.run_test_on_real_rasters(11, Resampling.cubic, False)

    def test_zoom_remote_12_resampling_cubic(self):
        self.run_test_on_real_rasters(12, Resampling.cubic, False)

    def test_zoom_remote_14_resampling_cubic(self):
        self.run_test_on_real_rasters(14, Resampling.cubic, False)

    def test_zoom_remote_15_resampling_cubic(self):
        self.run_test_on_real_rasters(15, Resampling.cubic, False)

    def test_zoom_remote_17_resampling_cubic(self):
        self.run_test_on_real_rasters(17, Resampling.cubic, False)

    def test_zoom_remote_18_resampling_cubic(self):
        self.run_test_on_real_rasters(18, Resampling.cubic, False)
