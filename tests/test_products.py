import unittest
import numpy as np
from telluric.products import ProductError

from common_for_tests import (
    make_test_raster, multi_raster_8b, multi_values_8b, expected_nir, expected_red, expected_green, sensor_bands_info,
    expected_blue, nir_value, red_value, green_value, blue_value, hyper_raster, hyper_raster_with_no_data,
    multi_raster_with_no_data, hyper_bands,
)

from telluric.products import (
    ProductsFactory, ProductGenerator, NDVI, EVI2, ENDVI, EXG, EXB, EXR, TrueColor, PRI, NDVI827, NRI, GNDVI, CCI,
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

    def test_products_order(self):
        keys = list(ProductsFactory.objects().keys())
        self.assertEqual(keys, ['rgbenhanced', 'truecolor', 'custom','singleband', 'ndvi', 'cci', 'gndvi',
                                'landcoverindex', 'ndvi750', 'ndvi827', 'npci', 'nri', 'ppr', 'pri', 'endvi',
                                'evi2', 'exb', 'exg', 'exr'])


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
        self.assertEqual(set(ProductsFactory.get_matchings(['red'], sensor_bands_info())), {'Custom', 'SingleBand'})


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


class TestTrueColor(unittest.TestCase):

    def test_true_color(self):
        raster = TrueColor().apply(sensor_bands_info(), multi_raster_8b())
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.num_bands, 3)
        # TODO what should I do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        for band in ['red', 'green', 'blue']:
            self.assertTrue((raster.bands_data(band) == multi_raster_8b().bands_data(band)).all())

    def test_TrueColor_product(self):
        product_generator = TrueColor()
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
        raster = TrueColor().apply(sensor_bands_info(), multi_raster_with_no_data())
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        # TODO what to do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, multi_raster_8b().height)
        self.assertEqual(raster.width, multi_raster_8b().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_TrueColor_product_for_macro(self):
        product_generator = TrueColor()
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
        expected_values = {'red': red_value, 'green': green_value, 'blue': blue_value}
        for band in ['red', 'green', 'blue']:
            self.assertTrue((product.raster.bands_data(band) == expected_values[band]).all())

    def test_truecolor_for_hyper_with_no_data(self):
        raster = TrueColor().apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.dtype, hyper_raster().dtype)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_fits_raster_bands_for_entire_hyper_band(self):
        true_color = ProductsFactory.get_object('truecolor')
        fits = true_color.fits_raster_bands(hyper_bands, sensor_bands_info())
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_RGB(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['red, green, blue'], sensor_bands_info())
        self.assertEqual(fits, False)

    def test_fits_raster_bands_false_for_part_of_the_hyper_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(hyper_bands[15:], sensor_bands_info())
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(hyper_bands[:6], sensor_bands_info())
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(hyper_bands[7:15], sensor_bands_info())
        self.assertEqual(fits, False)

    def test_fits_raster_bands_true_for_one_band_per_range(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_610'], sensor_bands_info())
        self.assertEqual(fits, True)

    def test_fits_raster_bands_true_neglecting_out_of_range_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_610', 'HC_300'], sensor_bands_info())
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_610', 'HC_580'], sensor_bands_info())
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_610', 'HC_700'], sensor_bands_info())
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_bands_in_range_hols_and_out_of_range(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_580'], sensor_bands_info())
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_512', 'HC_590'], sensor_bands_info())
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['HC_450', 'HC_550', 'HC_700'], sensor_bands_info())
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['HC_340', 'HC_550', 'HC_610'], sensor_bands_info())
        self.assertEqual(fits, False)

    def test_hs_true_color(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        raster = hs_true_color.apply(sensor_bands_info(), hyper_raster())
        self.assertEqual(raster.band_names, hs_true_color.output_bands)
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.dtype, hs_true_color.type)
        self.assertEqual(raster.height, hyper_raster().height)
        self.assertEqual(raster.width, hyper_raster().width)
        self.assertTrue((raster.bands_data('red') == red_value).all())
        self.assertTrue((raster.bands_data('green') == green_value).all())
        self.assertTrue((raster.bands_data('blue') == blue_value).all())

    def test_for_no_data_extended(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        raster = hs_true_color.apply(sensor_bands_info(), hyper_raster_with_no_data())
        self.assertEqual(raster.band_names, hs_true_color.output_bands)
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.dtype, hs_true_color.type)
        self.assertEqual(raster.height, hyper_raster_with_no_data().height)
        self.assertEqual(raster.width, hyper_raster_with_no_data().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[1, 1, 2])
        self.assertTrue(raster.image.mask[2, 1, 2])
        self.assertFalse(raster.image.mask[0, 1, 3])
        self.assertFalse(raster.image.mask[1, 1, 3])
        self.assertFalse(raster.image.mask[2, 1, 3])
        self.assertFalse(raster.image.mask[1, 0, 0])
        self.assertFalse(raster.image.mask[2, 0, 0])
        self.assertTrue(raster.image.mask[2, 2, 3])
        self.assertTrue(raster.image.mask[0, 2, 3])

    def test_HSTrueColor_product(self):
        product_generator = ProductsFactory.get_object('truecolor')
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        expected_bands_mapping = {'blue': expected_blue,
                                  'green': expected_green,
                                  'red': expected_red}

        for band in ['red', 'green', 'blue']:
            self.assertCountEqual(product.bands_mapping[band], expected_bands_mapping[band])


class TestSingleBand(unittest.TestCase):

    def test_single_band(self):
        raster = multi_raster_8b()
        product_raster = ProductsFactory.get_object('SingleBand', 'blue').apply(sensor_bands_info(), raster)
        self.assertEqual(product_raster, raster.limit_to_bands(['blue']))

    def test_object_vs_class(self):
        obj = ProductsFactory.get_object('SingleBand', 'blue')
        cls = ProductsFactory.get_class('SingleBand')
        self.assertEqual(type(obj), cls)

    def test_SingleBand_product(self):
        product_generator = ProductsFactory.get_object('SingleBand', 'blue')
        product = product_generator.apply(sensor_bands_info(), multi_raster_8b(), metadata=True)
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertEqual(product.bands_mapping, {'blue': ['blue']})

    def test_elements_in_bands(self):
        product_generator = ProductsFactory.get_object('SingleBand', 'HC_450')
        product = product_generator.apply(sensor_bands_info(), hyper_raster(), metadata=True)
        self.assertEqual(product.output_bands, ['HC_450'])


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


class TestRGBEnhanced(unittest.TestCase):
    enhanced_bands = ['red_enhanced', 'green_enhanced', 'blue_enhanced']

    @classmethod
    def enhanced_raster(cls):
        source_raster = make_test_raster(142, cls.enhanced_bands, dtype=np.uint8)
        for i, band in enumerate(cls.enhanced_bands):
            source_raster.image.data[i, :, :] = 2 * i + 100
        return source_raster

    def test_product(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        expected_bands_mapping = {
            'red': ['red_enhanced'],
            'green': ['green_enhanced'],
            'blue': ['blue_enhanced']
        }
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        product = product_generator.apply(sensor_bands_info(), self.enhanced_raster(), metadata=True)
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
        self.assertTrue((raster.bands_data('red') == 100.0).all())
        self.assertTrue((raster.bands_data('green') == 102.0).all())
        self.assertTrue((raster.bands_data('blue') == 104.0).all())

    def test_match_bands_for_legend(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        expected_bands_mapping = {
            'red': ['red_enhanced'],
            'green': ['green_enhanced'],
            'blue': ['blue_enhanced']
        }
        bands_mapping = product_generator.match_bands_for_legend(sensor_bands_info(), hyper_raster().band_names)
        self.assertCountEqual(bands_mapping, expected_bands_mapping)

    def test_product_fail_for_hyper_raster(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        mr = hyper_raster()
        with self.assertRaises(ProductError) as ex:
            product_generator.apply(sensor_bands_info(), mr)
        print(str(ex.exception))
        self.assertTrue('blue_enhanced' in str(ex.exception))
        self.assertTrue('red_enhanced' in str(ex.exception))
        self.assertTrue('green_enhanced' in str(ex.exception))


class TestCustomProduct(unittest.TestCase):
    def test_custom_red_plus_1(self):

        def foo(red):
            arr = red + 1
            return arr

        custom_product = ProductsFactory.get_object('custom', function=foo, required_bands=['red'],
                                                    output_bands=['band'])
        raster = make_test_raster(value=66, band_names=['red', 'green'])
        output = custom_product.apply(sensor_bands_info(), raster)
        self.assertCountEqual(output.shape[1:], raster.shape[1:])
        self.assertCountEqual(output.band_names, ['band'])
        self.assertTrue((output.image == 67).all())

    def test_custom_tow_bands_plus_1_and_2(self):

        def foo(red, green):
            arr1 = red + 1
            arr2 = green + 2
            arr = np.stack([arr1[0], arr2[0]])

            return arr

        custom_product = ProductsFactory.get_object('custom', function=foo, required_bands=['red', 'green'],
                                                    output_bands=['red', 'green'])
        raster = make_test_raster(value=66, band_names=['red', 'green', 'blue'])
        output = custom_product.apply(sensor_bands_info(), raster)
        self.assertCountEqual(output.shape[1:], raster.shape[1:])
        self.assertCountEqual(output.band_names, ['red', 'green'])
        self.assertTrue((output.bands_data('red') == 67).all())
        self.assertTrue((output.bands_data('green') == 68).all())

    def test_custom_product_preserve_mask(self):

        def foo(red, green):
            arr1 = red + 1
            arr2 = green + 2
            arr = np.stack([arr1[0], arr2[0]])
            return arr

        custom_product = ProductsFactory.get_object('custom', function=foo, required_bands=['red', 'green'],
                                                    output_bands=['red', 'green'])
        raster = multi_raster_with_no_data()
        output = custom_product.apply(sensor_bands_info(), raster)
        expected_mask = raster.bands_data('red').mask | raster.bands_data('green').mask
        self.assertTrue((output.image.mask == expected_mask).all())

        # raster.apply(sensor_bands_info(), 'custom_product', function=foo, required_bands=['red'], output_bands=['band'])

    def test_custom_tow_bands_plus_1_and_2_with_bands_mapping(self):

        def foo(red, green):
            arr1 = red + 1
            arr2 = green + 2
            arr = np.stack([arr1[0], arr2[0]])

            return arr

        bands_mapping = {'red': ['b1', 'b2'], 'green': ['b4', 'b5', 'b6']}
        custom_product = ProductsFactory.get_object('custom', function=foo, required_bands=['red', 'green'],
                                                    output_bands=['red', 'green'], bands_mapping=bands_mapping)
        raster = make_test_raster(value=66, band_names=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
        product = custom_product.apply(sensor_bands_info(), raster, metadata=True)
        output = product.raster
        self.assertCountEqual(output.shape[1:], raster.shape[1:])
        self.assertCountEqual(output.band_names, ['red', 'green'])
        self.assertTrue((output.bands_data('red') == 67).all())
        self.assertTrue((output.bands_data('green') == 68).all())
        self.assertDictEqual(product.bands_mapping, bands_mapping)
