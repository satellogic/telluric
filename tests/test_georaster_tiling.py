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

from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.crs import CRS

from telluric import GeoRaster2, GeoVector
from telluric.constants import WEB_MERCATOR_CRS, WGS84_CRS
from telluric.georaster import mercator_zoom_to_resolution, GeoRaster2Error, GeoRaster2IOError
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
    14: (9277, 6314, 14),
    15: (18554, 12628, 15),
    17: (74217, 50514, 17),
    18: (148434, 101028, 18)
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


class GeoRaster2windowContainedInRaster(TestCase):
    """GeoRaster2 _window_contained_in_raster tests."""

    def test_smaller_window_is_contained(self):
        raster = GeoRaster2(image=np.full((1, 20, 30), 12), affine=Affine.identity(), crs=WEB_MERCATOR_CRS)
        window = Window(10, 10, 10, 10)
        self.assertTrue(raster._window_contained_in_raster(window))

    def test_entire_raster_window_is_not_contained(self):
        raster = GeoRaster2(image=np.full((1, 20, 30), 12), affine=Affine.identity(), crs=WEB_MERCATOR_CRS)
        window = Window(0, 0, 30, 20)
        self.assertTrue(raster._window_contained_in_raster(window))

    def test_larger_window_is_not_contained(self):
        raster = GeoRaster2(image=np.full((1, 20, 30), 12), affine=Affine.identity(), crs=WEB_MERCATOR_CRS)
        window = Window(-1, -5, 35, 35)
        self.assertFalse(raster._window_contained_in_raster(window))

    def test_partial_intersecting_window_is_not_contained(self):
        raster = GeoRaster2(image=np.full((1, 20, 30), 12), affine=Affine.identity(), crs=WEB_MERCATOR_CRS)
        window = Window(-5, 5, 10, 25)
        self.assertFalse(raster._window_contained_in_raster(window))


class GeoRaster2TestGetTile(TestCase):
    """GeoRaster2 get tile tests."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(cls.temp_dir.name, 'test_raster.tif')
        if not os.path.isfile(path):
            cls.raster_for_test().save(path)
        cls.read_only_vgr = GeoRaster2.open(path)
        path = os.path.join(cls.temp_dir.name, 'small_test_raster.tif')
        if not os.path.isfile(path):
            cls.raster_small_for_test().save(path)
        cls.small_read_only_vgr = GeoRaster2.open(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @classmethod
    def raster_for_test(cls):
        # return GeoRaster2(np.random.uniform(0, 256, (3, 7823, 7417)),
        return GeoRaster2(np.random.uniform(0, 256, (3, 3911, 3708)),
                          affine=Affine(1.0000252884112817, 0.0, 2653750.345511198,
                                        0.0, -1.0000599330133702, 4594461.485763356),
                          crs={'init': 'epsg:3857'})

    @classmethod
    def raster_small_for_test(cls):
        return GeoRaster2(np.random.uniform(0, 256, (3, 391, 370)),
                          affine=Affine(1.0000252884112817, 0.0, 2653900.345511198,
                                        0.0, -1.0000599330133702, 4598361.485763356),
                          crs={'init': 'epsg:3857'})

    def read_only_virtual_geo_raster(self):
        return self.read_only_vgr

    def small_read_only_virtual_geo_raster(self):
        return self.small_read_only_vgr

    def test_geo_bounding_tile(self):
        gr = self.raster_for_test()
        gv = gr.footprint().reproject({'init': 'epsg:4326'})
        bounding_tile = mercantile.bounding_tile(*gv.get_shape(gv.crs).bounds)
        self.assertEqual(bounding_tile, (2319, 1578, 12))

    def test_fails_with_empty_raster_for_tile_out_of_raster_area(self):
        vr = self.read_only_virtual_geo_raster()
        r = vr.get_tile(16384, 16383, 15)
        self.assertTrue((r.image.data == 0).all())
        self.assertTrue((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_all_raster_in_a_single_tile(self):
        vr = self.read_only_virtual_geo_raster()
        r = vr.get_tile(1159, 789, 11)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_tile_for_different_zoom_levels(self):
        vr = self.read_only_virtual_geo_raster()
        for zoom in tiles:
            r = vr.get_tile(*tiles[zoom])
            self.assertFalse((r.image.data == 0).all())
            self.assertFalse((r.image.mask).all())
            self.assertEqual(r.image.shape, (3, 256, 256))

    def test_get_entire_all_raster(self):
        vr = self.small_read_only_virtual_geo_raster()
        r = vr.get_tile(37108, 25248, 16, blocksize=None)
        self.assertFalse((r.image.data == 0).all())
        self.assertFalse((r.image.mask).all())
        self.assertEqual(r.shape, (3, 612, 612))

    def test_fails_with_empty_raster_for_tile_out_of_raster_area_with_no_tile_size(self):
        vr = self.read_only_virtual_geo_raster()
        r = vr.get_tile(16384, 16383, 15, blocksize=None)
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


class GeoRasterCropTest(TestCase):
    metric_affine = Affine(1, 0.0, 2653750, 0.0, -1, 4594461)
    metric_crs = CRS({'init': 'epsg:3857'})
    geographic_affine = Affine(8.03258139076081e-06, 0.0, 23.83904185232179,
                               0.0, -8.03258139076081e-06, 38.10635414334363)
    geographic_crs = CRS({'init': 'epsg:4326'})

    def metric_raster(cls):
        return GeoRaster2(np.random.uniform(0, 256, (3, 3911, 3708)),
                          affine=cls.metric_affine,
                          crs=cls.metric_crs)

    def geographic_raster(cls):
        return GeoRaster2(np.random.uniform(0, 256, (3, 4147, 4147)),
                          affine=cls.geographic_affine,
                          crs=cls.geographic_crs)

    def test_crop_and_get_tile_do_without_resizing_the_same(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster2 = GeoRaster2.open(rf.name)
            tile15 = raster2.get_tile(*tiles[15], blocksize=None)
            # load the image data
            raster2.image
            cropped15 = raster2.crop(shape)
            self.assertEqual(tile15, cropped15)

    @window_data
    def test_crop_and_get_tile_do_the_same(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster2 = GeoRaster2.open(rf.name)
            tile15 = raster2.get_tile(*tiles[15])
            # load the image data
            raster2.image
            cropped15 = raster2.crop(shape, mercator_zoom_to_resolution[15])
            self.assertEqual(tile15, cropped15)

    @window_data
    def test_crop_and_get_tile_do_the_same_when_image_is_populated(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            tile15 = raster.get_tile(*tiles[15])
            raster._populate_from_rasterio_object(read_image=True)
            cropped_15 = raster.crop(shape, mercator_zoom_to_resolution[15])
            self.assertEqual(tile15, cropped_15)

    @window_data
    def test_crop_image_from_and_get_win_do_the_same_with_resize(self):
        bounds = (2, 3, 4, 5)
        win = rasterio.windows.Window(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        xsize = round((bounds[2] - bounds[0]) / 2)
        ysize = round((bounds[3] - bounds[1]) / 2)
        raster = self.metric_raster()

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
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile17 = raster.get_tile(*tiles[17])
            cropped_17 = raster.crop(shape, mercator_zoom_to_resolution[17])
            self.assertEqual(tile17, cropped_17)

    @framing
    def test_crop_and_get_tile_do_the_same_when_image_is_populated_first_mid_zoom(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile15 = raster.get_tile(*tiles[15])
            cropped_15 = raster.crop(shape, mercator_zoom_to_resolution[15])
            self.assertEqual(tile15, cropped_15)

    @framing
    def test_crop_and_get_tile_do_the_same_when_image_is_populated_first_for_low_zoom(self):
        coords = mercantile.xy_bounds(*tiles[11])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster._populate_from_rasterio_object(read_image=True)
            tile11 = raster.get_tile(*tiles[11])
            cropped_11 = raster.crop(shape, mercator_zoom_to_resolution[11])
            self.assertEqual(tile11, cropped_11)

    def test_crop_image_from_and_get_win_do_the_same_full_resolution(self):
        bounds = (200, 130, 400, 150)
        win = rasterio.windows.Window(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        raster = self.metric_raster()
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
        raster = self.metric_raster()
        raster.crop(shape, mercator_zoom_to_resolution[15])
        assert mock__crop.called_once

    @patch.object(GeoRaster2, 'get_window')
    def test_crop_use_get_window_for_a_not_loaded_image(self, mock_get_window):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        with NamedTemporaryFile(mode='w+b', suffix=".tif") as rf:
            raster.save(rf.name)
            raster = GeoRaster2.open(rf.name)
            raster.crop(shape, mercator_zoom_to_resolution[15])
            assert mock_get_window.called_once

    def test_crop_returns_full_resolution_as_default(self):
        coords = mercantile.xy_bounds(*tiles[17])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        _, win = raster._vector_to_raster_bounds(shape)
        cropped = raster.crop(shape)
        self.assertEqual(cropped.shape, (raster.num_bands, round(win.height), round(win.width)))
        self.assertEqual(cropped.affine[0], raster.affine[0])

    def test_memory_crop_returns_resized_resolution(self):
        coords = mercantile.xy_bounds(*tiles[15])
        shape = GeoVector(Polygon.from_bounds(*coords), WEB_MERCATOR_CRS)
        raster = self.metric_raster()
        cropped = raster.crop(shape, mercator_zoom_to_resolution[15])
        self.assertEqual(cropped.shape, (raster.num_bands, 256, 256))
        self.assertAlmostEqual(cropped.affine[0], mercator_zoom_to_resolution[15], 2)

    def test_geographic_crop(self):
        raster = self.geographic_raster()
        rhombus_on_image = Polygon([[0, 2], [1, 1], [2, 2], [1, 3]])  # in pixels
        rhombus_world = raster.to_world(rhombus_on_image)
        cropped = raster.crop(rhombus_world)
        r = raster[0:2, 1:3]
        assert cropped == r

    def test_geographic_crop_with_resize(self):
        coords = mercantile.xy_bounds(*tiles[17])
        raster = self.geographic_raster()
        vector = GeoVector(Polygon.from_bounds(*coords), crs=self.metric_crs).reproject(self.geographic_crs)
        cropped = raster.crop(vector, mercator_zoom_to_resolution[17])
        x_ex_res, y_ex_res = convert_resolution_from_meters_to_deg(
            self.metric_affine[6], mercator_zoom_to_resolution[17])
        self.assertAlmostEqual(cropped.affine[0], x_ex_res)
        self.assertAlmostEqual(abs(cropped.affine[4]), y_ex_res, 6)

    def test_crop_raises_error_for_impossible_transformation(self):
        raster = self.metric_raster()
        vector = GeoVector(Polygon.from_bounds(-180, -90, 180, 90), crs=self.geographic_crs)
        with self.assertRaises(GeoRaster2Error):
            raster.crop(vector)


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
        ], dtype=np.bool)

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
