import unittest
import pytest

import numpy as np
import affine
import telluric as tl
from telluric.constants import WGS84_CRS

from common_for_tests import make_test_raster, multi_raster_16b, multi_values_16b


from telluric.product_view import (
    ProductViewsFactory, ProductView,
    ColormapView, FirstBandGrayColormapView)


class TestProductViewsFactory(unittest.TestCase):

    def test_it_retrieves_the_right_product_views(self):
        self.assertTrue(ProductViewsFactory.get_object('cm-ylgn'))
        self.assertIsInstance(ProductViewsFactory.get_object('fb-cm-gray'), FirstBandGrayColormapView)

    def test_it_is_case_insensitive(self):
        self.assertIsInstance(ProductViewsFactory.get_object('fb-cM-GraY'), FirstBandGrayColormapView)

    def test_it_is_raises_key_error_when_productview_not_exists(self):
        self.assertRaises(KeyError, ProductViewsFactory.get_object, 'invalid_product')

    def test_product_view_is_not_instantiable(self):
        self.assertRaises(TypeError, ProductView, None)

    def test_colormap_view_is_not_instantiable(self):
        self.assertRaises(TypeError, ColormapView, None)


class TestProductView(unittest.TestCase):
    def test_to_dict(self):
        d = FirstBandGrayColormapView.to_dict()
        attributes = "name display_name description type output_bands required_bands".split()
        for att in attributes:
            self.assertIn(att, d)


class TestColormapView(unittest.TestCase):
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
        mask = heatmap.image.mask
        self.assertEqual(len(mask[mask == np.True_]), 3)


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
        self.assertTrue(np.array_equal(colormap.image.data[:, colormap.height - 1, colormap.width - 1],
                                       [255, 255, 255]))


class TestOneBanders(unittest.TestCase):

    def test_single_band(self):
        product = multi_raster_16b().limit_to_bands('red').astype(np.uint8)
        view = ProductViewsFactory.get_object('SingleBand').apply(product)
        self.assertEqual(view.shape, product.shape)
        self.assertEqual(view.dtype, product.dtype)

    def test_single_band_fails_on_multiband(self):
        with self.assertRaises(KeyError):
            multiband_product = multi_raster_16b()
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
