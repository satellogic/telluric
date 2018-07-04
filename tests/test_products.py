import unittest
import numpy as np
from telluric.products import ProductError

from common_for_tests import (
    make_test_raster, multi_raster_8b, multi_values_8b, expected_nir, expected_red, expected_green, sensor_bands_info,
    expected_blue, nir_value, red_value, green_value, blue_value, hyper_raster, hyper_raster_with_no_data,
    multi_raster_with_no_data,
)

from telluric.products import (
    ProductsFactory, ProductGenerator, NDVI, EVI2, ENDVI, EXG, EXB, EXR, PRI, NDVI827, NRI, GNDVI, CCI,
    NPCI, PPR, NDVI750, LandCoverIndex
)


class TestProductsFactory(unittest.TestCase):

    def test_it_retrieves_the_right_products(self):
        self.assertIsInstance(ProductsFactory.get_object('NDVI'), NDVI)
        self.assertIsInstance(ProductsFactory.get_object('ENDVI'), ENDVI)
        self.assertIsInstance(ProductsFactory.get_object('EVI2'), EVI2)

    def test_it_is_case_insensitive(self):
        self.assertIsInstance(ProductsFactory.get_object('ndvi'), NDVI)

    def test_it_is_raises_error_when_product_not_exists(self):
        self.assertRaises(KeyError, ProductsFactory.get_object, 'invalid_product')

    def test_product_is_not_instantiable(self):
        self.assertRaises(TypeError, ProductGenerator, None)


class TestProductGenerator(unittest.TestCase):

    def test_it_matches_all_bands(self):
        d = NDVI.to_dict()
        attributes = "name min max required_bands output_bands description default_view type display_name unit".split()
        for att in attributes:
            self.assertIn(att, d)

    def test_setting_minimal_value_for_zero_uint16(self):
        raster = make_test_raster(42, ['nir', 'red'])
        shape = (1, raster.height, raster.width)
        arr = np.ma.array(np.zeros(shape, dtype=np.uint16))
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue((output_arr.data == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_zero_uint8(self):
        raster = make_test_raster(42, ['nir', 'red'])
        shape = (1, raster.height, raster.width)
        arr = np.ma.array(np.zeros(shape, dtype=np.uint8))
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue((output_arr == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_zero_float32(self):
        raster = make_test_raster(0.42, ['nir', 'red'], dtype=np.float32)
        shape = (1, raster.height, raster.width)
        arr = np.ma.array(np.zeros(shape, dtype=np.float32))
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue((output_arr == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_nodata_uint16(self):
        raster = make_test_raster(42, ['nir', 'red'], dtype=np.uint16)
        array = raster.image.data
        mask = raster.image.mask
        mask[:, :, :] = True
        raster = raster.copy_with(image=np.ma.array(data=array, mask=mask))
        shape = (1, raster.height, raster.width)
        arr = np.ma.array(np.zeros(shape))
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue(output_arr.mask.all())

    def test_setting_minimal_value_for_nodata_float32(self):
        raster = make_test_raster(0.42, ['nir', 'red'], dtype=np.float32)
        array = raster.image.data
        mask = raster.image.mask
        mask[:, :, :] = True
        raster = raster.copy_with(image=np.ma.array(data=array, mask=mask))
        shape = (1, raster.height, raster.width)
        arr = np.ma.array(np.zeros(shape))
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue(output_arr.mask.all())


class TestBandsMatching(unittest.TestCase):

    def test_it_matches_all_bands(self):
        self.assertTrue(set(ProductsFactory.get_matchings(['blue', 'nir', 'red'],
                                                          sensor_bands_info())).issuperset(['NDVI', 'EVI2']))
        self.assertTrue(set(ProductsFactory.get_matchings(['nir', 'red'],
                                                          sensor_bands_info())).issuperset(['NDVI', 'EVI2']))
        self.assertNotIn('EXR', ProductsFactory.get_matchings(['nir', 'red'], sensor_bands_info()))
        self.assertEqual(ProductsFactory.get_matchings(['red'], sensor_bands_info()), [])


class TestNDVIStraight(unittest.TestCase):

    def test_ndvi(self):
        raster = NDVI().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        expected_value = (multi_values_8b['nir'] - multi_values_8b['red']) / (
            multi_values_8b['nir'] + multi_values_8b['red']
        )
        self.assertTrue((raster.image.data == expected_value).all())

    def test_NDVI_product_base(self):
        product_generator = NDVI()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'nir': ['nir'], 'red': ['red']})
        self.assertCountEqual(product.used_bands, ['nir', 'red'])
        self.assertEqual(product.default_view, product_generator.default_view)

    def test_for_no_data(self):
        raster = NDVI().apply(sensor_bands_info(), multi_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        self.assertFalse(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_NDVI_product_for_macro(self):
        product_generator = NDVI()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping, {'nir': expected_nir, 'red': expected_red})
        expected_value = (nir_value - red_value) / (nir_value + red_value)
        self.assertTrue((product.raster.image.data == expected_value).all())
        self.assertCountEqual(product.used_bands, expected_nir + expected_red)

    def test_for_hyper_with_no_data(self):
        raster = NDVI().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_NDVI_product_fail_for_hyper_with_missing_restricted_bands(self):
        product_generator = NDVI()
        mr = hyper_raster()
        restricted_bands = mr.band_names + ['HC_659']
        with self.assertRaises(ProductError) as ex:
            product_generator.apply(sensor_bands_info(), mr, bands_restriction=restricted_bands)
        self.assertEqual(str(ex.exception), "raster lacks restricted bands: HC_659")

    def test_NDVI_product_for_hyper_with_all_restricted_bands(self):
        product_generator = NDVI()
        mr = hyper_raster()
        restricted_bands = mr.band_names
        raster = product_generator.apply(sensor_bands_info(), mr, bands_restriction=restricted_bands)
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)


class TestEVI2(unittest.TestCase):

    def test_evi2(self):
        raster = EVI2().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        expected_value = 2.5 * (multi_values_8b['nir'] - multi_values_8b['red']) / (
            multi_values_8b['nir'] + 2.4 * multi_values_8b['red'] + 1
        )
        # expected_value = round(expected_value, 8)
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)

    def test_EVI2_product(self):
        product_generator = EVI2()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'nir': ['nir'], 'red': ['red']})

    def test_for_no_data(self):
        raster = EVI2().apply(sensor_bands_info(), multi_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        self.assertFalse(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_EVI2_product_for_macro(self):
        product_generator = EVI2()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping['nir'], expected_nir)
        self.assertCountEqual(product.bands_mapping['red'], expected_red)
        expected_value = 2.5 * (nir_value - red_value) / (nir_value + 2.4 * red_value + 1)
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_for_hyper_with_no_data(self):
        raster = EVI2().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestENDVI(unittest.TestCase):

    def test_endvi(self):
        raster = ENDVI().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        numerator = multi_values_8b['nir'] + multi_values_8b['green'] - 2 * multi_values_8b['blue']
        denominator = multi_values_8b['nir'] + multi_values_8b['green'] + 2 * multi_values_8b['blue']
        expected_value = numerator / denominator
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_ENDVI_product(self):
        product_generator = ENDVI()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'nir': ['nir'], 'green': ['green'], 'blue': ['blue']})

    def test_for_no_data(self):
        raster = ENDVI().apply(sensor_bands_info(), multi_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_ENDVI_product_for_macro(self):
        product_generator = ENDVI()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping['nir'], expected_nir)
        self.assertCountEqual(product.bands_mapping['green'], expected_green)
        self.assertCountEqual(product.bands_mapping['blue'], expected_blue)

        numerator = nir_value + green_value - 2 * blue_value
        denominator = nir_value + green_value + 2 * blue_value
        expected_value = numerator / denominator
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value, 5)
        self.assertTrue((product.raster.image.data == expected_value).all())

    def test_for_hyper_with_no_data(self):
        raster = ENDVI().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestExcessIndices(unittest.TestCase):

    def test_exg(self):
        raster = EXG().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXG.output_bands)
        self.assertEqual(raster.dtype, EXG.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        common_denominator = multi_values_8b['red'] + multi_values_8b['green'] + multi_values_8b['blue']
        r = multi_values_8b['red'] / common_denominator
        g = multi_values_8b['green'] / common_denominator
        b = multi_values_8b['blue'] / common_denominator
        expected_value = 2 * g - r - b
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_EXG_product(self):
        product_generator = EXG()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'red': ['red'], 'green': ['green'], 'blue': ['blue']})

    def test_for_no_data(self):
        raster = EXG().apply(sensor_bands_info(), multi_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_EXG_product_for_macro(self):
        product_generator = EXG()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping['red'], expected_red)
        self.assertCountEqual(product.bands_mapping['green'], expected_green)
        self.assertCountEqual(product.bands_mapping['blue'], expected_blue)
        common_denominator = red_value + green_value + blue_value
        r = red_value / common_denominator
        g = green_value / common_denominator
        b = blue_value / common_denominator
        expected_value = 2 * g - r - b
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exg_for_hyper_with_no_data(self):
        raster = EXG().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXG.output_bands)
        self.assertEqual(raster.dtype, EXG.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_exr(self):
        raster = EXR().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXR.output_bands)
        self.assertEqual(raster.dtype, EXR.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        common_denominator = multi_values_8b['red'] + multi_values_8b['green'] + multi_values_8b['blue']
        r = multi_values_8b['red'] / common_denominator
        g = multi_values_8b['green'] / common_denominator
        expected_value = 1.4 * r - g
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_EXR_product(self):
        product_generator = EXR()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'red': ['red'], 'green': ['green'], 'blue': ['blue']})

    def test_EXR_product_for_macro(self):
        product_generator = EXR()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping['red'], expected_red)
        self.assertCountEqual(product.bands_mapping['green'], expected_green)
        self.assertCountEqual(product.bands_mapping['blue'], expected_blue)
        common_denominator = red_value + green_value + blue_value
        r = red_value / common_denominator
        g = green_value / common_denominator
        expected_value = 1.4 * r - g
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exr_for_hyper_with_no_data(self):
        raster = EXR().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXR.output_bands)
        self.assertEqual(raster.dtype, EXR.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_exb(self):
        raster = EXB().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXB.output_bands)
        self.assertEqual(raster.dtype, EXB.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        common_denominator = multi_values_8b['red'] + multi_values_8b['green'] + multi_values_8b['blue']
        g = multi_values_8b['green'] / common_denominator
        b = multi_values_8b['blue'] / common_denominator
        expected_value = 1.4 * b - g
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)

    def test_EXB_product(self):
        product_generator = EXB()
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'red': ['red'], 'green': ['green'], 'blue': ['blue']})

    def test_EXB_product_for_macro(self):
        product_generator = EXB()
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping['red'], expected_red)
        self.assertCountEqual(product.bands_mapping['green'], expected_green)
        self.assertCountEqual(product.bands_mapping['blue'], expected_blue)
        common_denominator = red_value + green_value + blue_value
        g = green_value / common_denominator
        b = blue_value / common_denominator
        expected_value = 1.4 * b - g
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exb_for_hyper_with_no_data(self):
        raster = EXB().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXB.output_bands)
        self.assertEqual(raster.dtype, EXB.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class BaseTestMacroProduct(unittest.TestCase):

    def do_test_product_for_macro(self, product_generator, expected_bands_mapping,
                                  expected_bands_names, expected_value):
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        for band in expected_bands_mapping:
            self.assertCountEqual(product.bands_mapping[band], expected_bands_mapping[band])
        self.assertTrue((product.raster.image.data == expected_value).all())
        self.assertCountEqual(product.used_bands, expected_bands_names)
        return product

    def do_test_for_hyper_with_no_data(self, product_generator):
        product = product_generator.apply(sensor_bands_info(), hyper_raster_with_no_data(), metadata=True)
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, product_generator.output_bands)
        self.assertEqual(raster.dtype, product_generator.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        return product


class TestPRI(BaseTestMacroProduct):

    def test_PRI_product_for_macro(self):
        expected_bands_mapping = {'R530': ['HC_530'], 'R570': ['HC_570']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (108.0 - 104.0) / (108.0 + 104.0)
        self.do_test_product_for_macro(PRI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(PRI())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestNDVI827(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R827': ['HC_830'], 'R690': ['HC_690']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (138.0 - 120.0) / (138.0 + 120.0)
        self.do_test_product_for_macro(NDVI827(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(NDVI827())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestNRI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R570': ['HC_570'], 'R670': ['HC_670']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (108.0 - 116.0) / (108.0 + 116.0)
        self.do_test_product_for_macro(NRI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(NRI())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestGNDVI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R750': ['HC_750'], 'R550': ['HC_550']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (132.0 - 106.0) / (132.0 + 106.0)
        self.do_test_product_for_macro(GNDVI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(GNDVI())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestCCI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R530': ['HC_530'], 'R670': ['HC_670']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (104.0 - 116.0) / (104.0 + 116.0)
        self.do_test_product_for_macro(CCI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(CCI())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestNPCI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R582': ['HC_580'], 'R450': ['HC_450']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (110.0 - 100.0) / (110.0 + 100.0)
        self.do_test_product_for_macro(NPCI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(NPCI())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestPPR(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R550': ['HC_550'], 'R450': ['HC_450']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (106.0 - 100.0) / (106.0 + 100.0)
        self.do_test_product_for_macro(PPR(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(PPR())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestNDVI750(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R750': ['HC_750'], 'R700': ['HC_700']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (132.0 - 122.0) / (132.0 + 122.0)
        self.do_test_product_for_macro(NDVI750(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_hyper_with_no_data(self):
        product = self.do_test_for_hyper_with_no_data(NDVI750())
        raster = product.raster
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestLandCoverIndex(unittest.TestCase):

    def test_product_for_macro(self):
        product_generator = LandCoverIndex()
        expected_bands_mapping = {
            'red': ['HC_830'],
            'green': ['HC_690'],
            'blue': ['HC_550']
        }
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping, expected_bands_mapping)
        self.assertCountEqual(product.used_bands, expected_bands_names)
        raster = product.raster
        self.assertTrue((raster.bands_data('red') == 138.0).all())
        self.assertTrue((raster.bands_data('green') == 120.0).all())
        self.assertTrue((raster.bands_data('blue') == 106.0).all())

    def test_match_bands_for_legend(self):
        product_generator = LandCoverIndex()
        expected_bands_mapping = {
            'red': ['HC_830'],
            'green': ['HC_690'],
            'blue': ['HC_550']
        }
        bands_mapping = product_generator.match_bands_for_legend(sensor_bands_info(), hyper_raster().band_names)
        self.assertCountEqual(bands_mapping, expected_bands_mapping)
