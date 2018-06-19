import unittest
import pytest
import numpy as np
import affine
import telluric as tl
from telluric.constants import WGS84_CRS, WEB_MERCATOR_CRS
# import rastile.utils.geography
from telluric.product_view import (
    ProductViewsFactory, SingleBand, Grayscale, TrueColor, FalseColor, ProductView, BandsComposer,
    ColormapView, FirstBandGrayColormapView)


def make_test_raster(value=0, band_names=[], height=3, width=4, dtype=np.uint16,
                     crs=WEB_MERCATOR_CRS, affine=affine.Affine.identity()):
    shape = [len(band_names), height, width]
    array = np.full(shape, value, dtype=dtype)
    mask = np.full(shape, False, dtype=np.bool)
    image = np.ma.array(data=array, mask=mask)
    raster = tl.GeoRaster2(image=image, affine=affine, crs=crs, band_names=band_names)
    return raster


micro_values = {
    'green': 5500,
    'red': 7700,
    'nir': 10000,
    'blue': 11800
}


def micro_raster():
    source_raster = make_test_raster(4200, ['green', 'red', 'nir', 'blue'], dtype=np.uint16)
    array = source_raster.image.data
    array[0, :, :] = micro_values['green']
    array[1, :, :] = micro_values['red']
    array[2, :, :] = micro_values['nir']
    array[3, :, :] = micro_values['blue']
    return source_raster.copy_with(image=array)


class TestProductViewsFactory(unittest.TestCase):

    def test_it_retrieves_the_right_product_views(self):
        self.assertTrue(ProductViewsFactory.get_object('cm-ylgn'))
        self.assertIsInstance(ProductViewsFactory.get_object('singleband'), SingleBand)
        self.assertIsInstance(ProductViewsFactory.get_object('grayscale'), Grayscale)
        self.assertIsInstance(ProductViewsFactory.get_object('truecolor'), TrueColor)
        self.assertIsInstance(ProductViewsFactory.get_object('falsecolor'), FalseColor)
        self.assertIsInstance(ProductViewsFactory.get_object('fb-cm-gray'), FirstBandGrayColormapView)

    def test_it_is_case_insensitive(self):
        self.assertIsInstance(ProductViewsFactory.get_object('TrUeColOr'), TrueColor)

    def test_it_is_raises_key_error_whene_productview_not_exists(self):
        self.assertRaises(KeyError, ProductViewsFactory.get_object, 'invalid_product')

    def test_produc_view_is_not_instaciable(self):
        self.assertRaises(TypeError, ProductView, None)

    def test_colormap_view_is_not_instaciable(self):
        self.assertRaises(TypeError, ColormapView, None)

    def test_band_composer_is_not_instaciable(self):
        self.assertRaises(TypeError, BandsComposer, None)


class TestProductView(unittest.TestCase):
    def test_it_matches_all_bands(self):
        d = Grayscale.to_dict()
        attributes = "name display_name description type output_bands required_bands".split()
        for att in attributes:
            self.assertIn(att, d)


class TestBandsMatching(unittest.TestCase):
    def test_it_matches_all_bands(self):
        self.assertTrue(set(ProductViewsFactory.get_matchings(
            ['blue', 'green', 'red'])).issuperset(['Grayscale', 'TrueColor']))
        self.assertNotIn('FalseColor', ProductViewsFactory.get_matchings(
            ['blue', 'green', 'red']))
        self.assertTrue(set(ProductViewsFactory.get_matchings(['nir', 'red', 'green'])).issuperset(['FalseColor']))
        self.assertTrue(len(ProductViewsFactory.get_matchings(['red'])) > 0)


class TestCloromapView(unittest.TestCase):
    def test_heatmap(self):
        raster = tl.GeoRaster2(image=np.array(range(256), dtype=np.uint8).reshape((1, 16, 16)),
                               band_names=['red'], 
                               crs=WGS84_CRS,
                               affine=affine.Affine(2, 0, 0, 0, 1, 0))

        renderer = ProductViewsFactory.get_object('cm-jet')
        heatmap = renderer.apply(raster)
        self.assertTrue(np.array_equal(heatmap.band_names, ['red', 'green', 'blue']))
        self.assertTrue(np.array_equal(heatmap.image.data[:, 0, 0], [0, 0, 0]))  # nodata remains nodata
        self.assertTrue(np.array_equal(heatmap.image.mask[:, 0, 0], [True, True, True]))  # nodata remains nodata
        self.assertTrue(np.array_equal(heatmap.image.data[:, 0, 1], [0, 0, 127]))  # blue
        self.assertTrue(np.array_equal(heatmap.image.mask[:, 0, 1], [False, False, False]))  # blue
        self.assertTrue(np.array_equal(heatmap.image.data[:, heatmap.height - 1, heatmap.width - 1],
                                       [127, 0, 0]))  # red
        self.assertTrue(np.array_equal(heatmap.image.mask[:, heatmap.height - 1, heatmap.width - 1],
                                       [False, False, False]))  # red

    def test_with_range(self):
        raster = tl.GeoRaster2(image=np.array(range(256), dtype=np.uint8).reshape((1, 16, 16)),
                               band_names=['red'], 
                               crs=WGS84_CRS,
                               affine=affine.Affine(2, 0, 0, 0, 1, 0))

        renderer = ProductViewsFactory.get_object('cm-jet')
        heatmap = renderer.apply(raster, vmin=-1, vmax=1)
        self.assertTrue(np.array_equal(heatmap.band_names, ['red', 'green', 'blue']))
        self.assertTrue(np.array_equal(heatmap.image.data[:, 0, 0], [0, 0, 0]))  # nodata remains nodata
        self.assertTrue(np.array_equal(heatmap.image.mask[:, 0, 0], [True, True, True]))  # nodata remains nodata
        self.assertTrue((heatmap.image.data[0, heatmap.image.data[0] > 0] == 127).all())
        self.assertTrue((heatmap.image.data[1, heatmap.image.data[1] > 0] == 0).all())
        self.assertTrue((heatmap.image.data[2, heatmap.image.data[2] > 0] == 0).all())
        self.assertEqual(len(heatmap.image.mask == True), 3)


class TestFirstBandGrayColormapView(unittest.TestCase):
    def test_view(self):
        raster = tl.GeoRaster2(image=np.array(range(512), dtype=np.uint8).reshape((2, 16, 16)),
                               band_names=['main', 'else'], 
                               crs=WGS84_CRS,
                               affine=affine.Affine(2, 0, 0, 0, 1, 0))

        renderer = ProductViewsFactory.get_object('fb-cm-gray')
        colormap = renderer.apply(raster)
        self.assertTrue(np.array_equal(colormap.band_names, ['red', 'green', 'blue']))
        self.assertTrue(np.array_equal(colormap.image.data[:, 0, 0], [0, 0, 0]))  # nodata remains nodata
        self.assertTrue(np.array_equal(colormap.image.mask[:, 0, 0], [True, True, True]))  # nodata remains nodata
        self.assertTrue(np.array_equal(colormap.image.data[:, 0, 1], [0, 0, 0]))
        self.assertTrue(np.array_equal(colormap.image.data[:, colormap.height - 1, colormap.width - 1], [255, 255, 255]))


def to_uint8(val):
    return int(val / 257)


class TestBandComposer(unittest.TestCase):
    def test_true_color_raises_on_no_fiting_requirements(self):
        raster = make_test_raster(4200, ['green', 'red'], dtype=np.uint16)
        with self.assertRaises(KeyError):
            raster = TrueColor().apply(raster)

    def test_true_color(self):
        raster = TrueColor().apply(micro_raster())
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, ['red', 'green', 'blue'])
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertTrue((raster.image.data[0, :, :] == to_uint8(micro_values['red'])).all())
        self.assertTrue((raster.image.data[1, :, :] == to_uint8(micro_values['green'])).all())
        self.assertTrue((raster.image.data[2, :, :] == to_uint8(micro_values['blue'])).all())

    def test_false_color(self):
        raster = FalseColor().apply(micro_raster())
        self.assertTrue((raster.image.data[0, :, :] == to_uint8(micro_values['nir'])).all())
        self.assertTrue((raster.image.data[1, :, :] == to_uint8(micro_values['red'])).all())
        self.assertTrue((raster.image.data[2, :, :] == to_uint8(micro_values['green'])).all())

    def test_false_color_raises_on_no_fiting_requirements(self):
        raster = make_test_raster(4200, ['green', 'blue'], dtype=np.uint16)
        with self.assertRaises(KeyError):
            raster = FalseColor().apply(raster)

    def test_grayscale(self):
        raster = ProductViewsFactory.get_object('Grayscale').apply(micro_raster())
        self.assertEqual(raster.band_names, ['grayscale'])
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        expected_value = 0.2989 * micro_values['red'] + 0.5870 * micro_values['green'] + 0.1140 * micro_values['blue']
        self.assertEqual(raster.image.data[0, 0, 0], to_uint8(expected_value))

    def test_grayscale_raises_on_no_fiting_requirements(self):
        raster = make_test_raster(4200, ['blue', 'red'], dtype=np.uint16)
        with self.assertRaises(KeyError):
            raster = Grayscale().apply(raster)


class TestOneBanders(unittest.TestCase):

    def test_single_band(self):
        product = micro_raster().limit_to_bands('red').astype(np.uint8)
        view = ProductViewsFactory.get_object('SingleBand').apply(product)
        self.assertEqual(view.shape, product.shape)
        self.assertEqual(view.dtype, product.dtype)

    def test_single_band_fails_on_multiband(self):
        with self.assertRaises(KeyError):
            multiband_product = micro_raster()
            ProductViewsFactory.get_object('SingleBand').apply(multiband_product)

    @pytest.mark.xfail(reason="waiting for implementing astype for non integers")
    def test_magnifying_glass_float32(self):
        product = make_test_raster(0.1, ['red'], dtype=np.float32)
        mag_glass = ProductViewsFactory.get_object('magnifyingglass')
        view = mag_glass.apply(product)
        self.assertEqual(view.shape, product.shape)
        self.assertEqual(view.dtype, mag_glass.type)
        self.assertTrue((view.image.data == 110).all())

    def test_magnifying_glass_uint16(self):
        product = make_test_raster(640, ['red'], dtype=np.uint16)
        mag_glass = ProductViewsFactory.get_object('magnifyingglass')
        view = mag_glass.apply(product)
        self.assertEqual(view.shape, product.shape)
        self.assertEqual(view.dtype, mag_glass.type)
        self.assertTrue((view.image.data == 249).all())

    def test_magnifying_glass_uint8(self):
        product = make_test_raster(1, ['red'], dtype=np.uint8)
        mag_glass = ProductViewsFactory.get_object('magnifyingglass')
        view = mag_glass.apply(product)
        self.assertEqual(view.shape, product.shape)
        self.assertEqual(view.dtype, mag_glass.type)
        self.assertTrue((view.image.data == 200).all())
