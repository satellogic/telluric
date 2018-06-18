import unittest
import numpy as np
import copy

from affine import Affine
import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS
from telluric.products import ProductError

from telluric.products import (
    ProductsFactory, ProductGenerator, NDVI, EVI2, ENDVI, EXG, EXB, EXR, TrueColor, PRI, NDVI827, NRI, GNDVI, CCI,
    NPCI, PPR, NDVI750, LandCoverIndex
)


def make_test_raster(value=0, band_names=[], height=3, width=4, dtype=np.uint16,
                     crs=WEB_MERCATOR_CRS, affine=Affine.identity()):
    shape = [len(band_names), height, width]
    array = np.full(shape, value, dtype=dtype)
    mask = np.full(shape, False, dtype=np.bool)
    image = np.ma.array(data=array, mask=mask)
    raster = tl.GeoRaster2(image=image, affine=affine, crs=crs, band_names=band_names)
    return raster


def sensor_bands_info():
    return copy.deepcopy({
        'blue': {'min': 400, 'max': 510},
        'green': {'min': 515, 'max': 580},
        'pan': {'min': 400, 'max': 750},
        'red': {'min': 590, 'max': 690},
        'nir': {'min': 750, 'max': 900},
        'HC_450': {'min': 445, 'max': 456},
        'HC_500': {'min': 493, 'max': 510},
        'HC_530': {'min': 521, 'max': 538},
        'HC_550': {'min': 540, 'max': 559},
        'HC_570': {'min': 560, 'max': 579},
        'HC_580': {'min': 572, 'max': 592},
        'HC_595': {'min': 583, 'max': 605},
        'HC_610': {'min': 596, 'max': 619},
        'HC_659': {'min': 646, 'max': 671},
        'HC_670': {'min': 657, 'max': 683},
        'HC_680': {'min': 666, 'max': 693},
        'HC_690': {'min': 676, 'max': 703},
        'HC_700': {'min': 686, 'max': 714},
        'HC_710': {'min': 696, 'max': 724},
        'HC_720': {'min': 705, 'max': 734},
        'HC_730': {'min': 714, 'max': 744},
        'HC_740': {'min': 725, 'max': 755},
        'HC_750': {'min': 735, 'max': 765},
        'HC_760': {'min': 744, 'max': 775},
        'HC_770': {'min': 753, 'max': 785},
        'HC_830': {'min': 809, 'max': 844},
    })

macro_wavelengths = (450, 500, 530, 550, 570, 580, 595, 610, 670, 680,
                     690, 700, 710, 720, 730, 740, 750, 760, 770, 830)

micro_values = {
    'green': 55,
    'red': 77,
    'nir': 100,
    'blue': 118
}

expected_nir = ['HC_770', 'HC_830']
expected_red = ['HC_670', 'HC_610']
expected_green = ['HC_570', 'HC_550', 'HC_530']
expected_blue = ['HC_450', 'HC_500']

# calculated micro values
nir_value = 137    # nir values 138 136 avg 137
red_value = 115    # red values 116 114 avg 115
green_value = 106  # green values 104 106 108 avg 106
blue_value = 101   # blue values 100 102 avg 101


def micro_raster():
    source_raster = make_test_raster(4200, ['green', 'red', 'nir', 'blue'], dtype=np.uint16)
    array = source_raster.image.data
    array[0, :, :] = micro_values['green']
    array[1, :, :] = micro_values['red']
    array[2, :, :] = micro_values['nir']
    array[3, :, :] = micro_values['blue']
    return source_raster.copy_with(image=array)


def micro_raster_with_no_data():
    source_raster = micro_raster()
    source_raster.image.mask[0, 1, 2] = True
    source_raster.image.mask[1, 2, 3] = True
    source_raster.image.mask[2, 0, 0] = True
    source_raster.image.mask[3, 1, 3] = True
    return source_raster


macro_bands = ["HC_%i" % wl for wl in macro_wavelengths]


def macro_raster():
    source_raster = make_test_raster(142, macro_bands, dtype=np.uint8)
    for i, band in enumerate(macro_bands):
        source_raster.image.data[i, :, :] = 2 * i + 100
    return source_raster


def macro_raster_with_no_data():
    source_raster = macro_raster()
    source_raster.image.mask[:, 1, 2] = True
    source_raster.image.mask[2, 2, 3] = True
    source_raster.image.mask[5, 0, 0] = True
    source_raster.image.mask[14, 1, 3] = True
    source_raster.image.mask[15, 1, 3] = True
    return source_raster


class TestProductsFuctory(unittest.TestCase):

    def test_it_retrieves_the_right_products(self):
        self.assertIsInstance(ProductsFactory.get_object('NDVI'), NDVI)
        self.assertIsInstance(ProductsFactory.get_object('ENDVI'), ENDVI)
        self.assertIsInstance(ProductsFactory.get_object('EVI2'), EVI2)

    def test_it_is_case_insensitive(self):
        self.assertIsInstance(ProductsFactory.get_object('ndvi'), NDVI)

    def test_it_is_raises_error_whene_product_not_exists(self):
        self.assertRaises(KeyError, ProductsFactory.get_object, 'invalid_product')

    def test_produc_is_not_instaciable(self):
        self.assertRaises(TypeError, ProductGenerator, None)

    def test_products_order(self):
        keys = list(ProductsFactory.objects().keys())
        self.assertEqual(keys, ['rgbenhanced', 'truecolor', 'singleband', 'ndvi', 'cci', 'gndvi',
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
        self.assertTrue(set(ProductsFactory.get_matchings(sensor_bands_info(),
                        ['blue', 'nir', 'red'])).issuperset(['NDVI', 'EVI2']))
        self.assertTrue(set(ProductsFactory.get_matchings(sensor_bands_info(),
                        ['nir', 'red'])).issuperset(['NDVI', 'EVI2']))
        self.assertNotIn('EXR', ProductsFactory.get_matchings(sensor_bands_info(), ['nir', 'red']))
        self.assertEqual(set(ProductsFactory.get_matchings(sensor_bands_info(), ['red'])), {'SingleBand'})


class TestNDVIStraite(unittest.TestCase):

    def test_ndvi(self):
        raster = NDVI().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        expected_value = (micro_values['nir'] - micro_values['red']) / (
            micro_values['nir'] + micro_values['red']
        )
        self.assertTrue((raster.image.data == expected_value).all())

    def test_NDVI_product_base(self):
        product_generator = NDVI()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        raster = NDVI().apply(sensor_bands_info(), micro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertFalse(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_NDVI_product_for_macro(self):
        product_generator = NDVI()
        # import pdb; pdb.set_trace()
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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

    def test_for_macro_with_no_data(self):
        raster = NDVI().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_NDVI_product_fail_for_macro_with_missing_restricted_bands(self):
        product_generator = NDVI()
        mr = macro_raster()
        restricted_bands = mr.band_names + ['HC_659']
        with self.assertRaises(ProductError) as ex:
            product_generator.apply(sensor_bands_info(), mr, bands_restriction=restricted_bands)
        self.assertEqual(str(ex.exception), "raster lacks restricted bands: HC_659")

    def test_NDVI_product_for_macro_with_all_restricted_bands(self):
        product_generator = NDVI()
        mr = macro_raster()
        restricted_bands = mr.band_names
        raster = product_generator.apply(sensor_bands_info(), mr, bands_restriction=restricted_bands)
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)


class TestEVI2(unittest.TestCase):

    def test_evi2(self):
        raster = EVI2().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        expected_value = 2.5 * (micro_values['nir'] - micro_values['red']) / (
            micro_values['nir'] + 2.4 * micro_values['red'] + 1
            )
        # expected_value = round(expected_value, 8)
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)

    def test_EVI2_product(self):
        product_generator = EVI2()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        raster = EVI2().apply(sensor_bands_info(), micro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertFalse(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_EVI2_product_for_macro(self):
        product_generator = EVI2()
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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

    def test_for_macro_with_no_data(self):
        raster = EVI2().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestENDVI(unittest.TestCase):

    def test_endvi(self):
        raster = ENDVI().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        numerator = micro_values['nir'] + micro_values['green'] - 2 * micro_values['blue']
        denominator = micro_values['nir'] + micro_values['green'] + 2 * micro_values['blue']
        expected_value = numerator / denominator
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_ENDVI_product(self):
        product_generator = ENDVI()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        raster = ENDVI().apply(sensor_bands_info(), micro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertFalse(raster.image.mask[0, 2, 3])
        self.assertTrue(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_ENDVI_product_for_macro(self):
        product_generator = ENDVI()
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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

    def test_for_macro_with_no_data(self):
        raster = ENDVI().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestExcessIndices(unittest.TestCase):

    def test_exg(self):
        raster = EXG().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXG.output_bands)
        self.assertEqual(raster.dtype, EXG.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        comon_denominator = micro_values['red'] + micro_values['green'] + micro_values['blue']
        r = micro_values['red'] / comon_denominator
        g = micro_values['green'] / comon_denominator
        b = micro_values['blue'] / comon_denominator
        expected_value = 2 * g - r - b
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_EXG_product(self):
        product_generator = EXG()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        raster = EXG().apply(sensor_bands_info(), micro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_EXG_product_for_macro(self):
        product_generator = EXG()
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
        comon_denominator = red_value + green_value + blue_value
        r = red_value / comon_denominator
        g = green_value / comon_denominator
        b = blue_value / comon_denominator
        expected_value = 2 * g - r - b
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exg_for_macro_with_no_data(self):
        raster = EXG().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXG.output_bands)
        self.assertEqual(raster.dtype, EXG.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_exr(self):
        raster = EXR().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXR.output_bands)
        self.assertEqual(raster.dtype, EXR.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        comon_denominator = micro_values['red'] + micro_values['green'] + micro_values['blue']
        r = micro_values['red'] / comon_denominator
        g = micro_values['green'] / comon_denominator
        expected_value = 1.4 * r - g
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)
        self.assertTrue((raster.image.data == expected_value).all())

    def test_EXR_product(self):
        product_generator = EXR()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
        comon_denominator = red_value + green_value + blue_value
        r = red_value / comon_denominator
        g = green_value / comon_denominator
        expected_value = 1.4 * r - g
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exr_for_macro_with_no_data(self):
        raster = EXR().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXR.output_bands)
        self.assertEqual(raster.dtype, EXR.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_exb(self):
        raster = EXB().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXB.output_bands)
        self.assertEqual(raster.dtype, EXB.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        comon_denominator = micro_values['red'] + micro_values['green'] + micro_values['blue']
        g = micro_values['green'] / comon_denominator
        b = micro_values['blue'] / comon_denominator
        expected_value = 1.4 * b - g
        self.assertAlmostEqual(raster.image.data[0, 0, 0], expected_value)

    def test_EXB_product(self):
        product_generator = EXB()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
        comon_denominator = red_value + green_value + blue_value
        g = green_value / comon_denominator
        b = blue_value / comon_denominator
        expected_value = 1.4 * b - g
        self.assertAlmostEqual(product.raster.image.data[0, 0, 0], expected_value)

    def test_exb_for_macro_with_no_data(self):
        raster = EXB().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXB.output_bands)
        self.assertEqual(raster.dtype, EXB.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])


class TestTrueColor(unittest.TestCase):

    def test_true_color(self):
        raster = TrueColor().apply(sensor_bands_info(), micro_raster())
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.num_bands, 3)
        # TODO what should I do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        for band in ['red', 'green', 'blue']:
            self.assertTrue((raster.band(band) == micro_raster().band(band)).all())

    def test_TrueColor_product(self):
        product_generator = TrueColor()
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        raster = TrueColor().apply(sensor_bands_info(), micro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        # TODO what to do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertTrue(raster.image.mask[0, 1, 3])

    def test_TrueColor_product_for_macro(self):
        product_generator = TrueColor()
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
            self.assertTrue((product.raster.band(band) == expected_values[band]).all())

    def test_truecolor_for_macro_with_no_data(self):
        raster = TrueColor().apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.dtype, macro_raster().dtype)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.mask[0, 1, 2])
        self.assertTrue(raster.image.mask[0, 2, 3])
        self.assertFalse(raster.image.mask[0, 0, 0])
        self.assertFalse(raster.image.mask[0, 1, 3])

    def test_fits_raster_bands_for_entire_macro_band(self):
        true_color = ProductsFactory.get_object('truecolor')
        fits = true_color.fits_raster_bands(sensor_bands_info(), macro_bands)
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_RGB(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['red, green, blue'])
        self.assertEqual(fits, False)

    def test_fits_raster_bands_false_for_part_of_the_macro_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), macro_bands[15:])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), macro_bands[:6])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), macro_bands[7:15])
        self.assertEqual(fits, False)

    def test_fits_raster_bands_true_for_one_band_per_range(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_610'])
        self.assertEqual(fits, True)

    def test_fits_raster_bands_true_niglecting_out_of_range_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_610', 'HC_300'])
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_610', 'HC_580'])
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_610', 'HC_700'])
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_bands_in_range_hols_and_out_of_renge(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_580'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_512', 'HC_590'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_450', 'HC_550', 'HC_700'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(sensor_bands_info(), ['HC_340', 'HC_550', 'HC_610'])
        self.assertEqual(fits, False)

    def test_hs_true_color(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        raster = hs_true_color.apply(sensor_bands_info(), macro_raster())
        self.assertEqual(raster.band_names, hs_true_color.output_bands)
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.dtype, hs_true_color.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue((raster.band('red') == red_value).all())
        self.assertTrue((raster.band('green') == green_value).all())
        self.assertTrue((raster.band('blue') == blue_value).all())

    def test_for_no_data_extended(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        raster = hs_true_color.apply(sensor_bands_info(), macro_raster_with_no_data())
        self.assertEqual(raster.band_names, hs_true_color.output_bands)
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.dtype, hs_true_color.type)
        self.assertEqual(raster.height, macro_raster_with_no_data().height)
        self.assertEqual(raster.width, macro_raster_with_no_data().width)
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
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
        raster = micro_raster()
        product_raster = ProductsFactory.get_object('SingleBand', 'blue').apply(sensor_bands_info(), raster)
        self.assertEqual(product_raster, raster.limit_to_bands(['blue']))

    def test_object_vs_class(self):
        obj = ProductsFactory.get_object('SingleBand', 'blue')
        cls = ProductsFactory.get_class('SingleBand')
        self.assertEqual(type(obj), cls)

    def test_SingleBand_product(self):
        product_generator = ProductsFactory.get_object('SingleBand', 'blue')
        product = product_generator.apply(sensor_bands_info(), micro_raster(), metadata=True)
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
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
        self.assertEqual(product.output_bands, ['HC_450'])


class BaseTestMacroProduct(unittest.TestCase):

    def do_test_product_for_macro(self, product_generator, expected_bands_mapping,
                                  expected_bands_names, expected_value):
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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

    def do_test_for_macro_with_no_data(self, product_generator):
        product = product_generator.apply(sensor_bands_info(), macro_raster_with_no_data(), metadata=True)
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, product_generator.output_bands)
        self.assertEqual(raster.dtype, product_generator.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        return product


class TestPRI(BaseTestMacroProduct):

    def test_PRI_product_for_macro(self):
        expected_bands_mapping = {'R530': ['HC_530'], 'R570': ['HC_570']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (108.0 - 104.0) / (108.0 + 104.0)
        self.do_test_product_for_macro(PRI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(PRI())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NDVI827())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NRI())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(GNDVI())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(CCI())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NPCI())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(PPR())
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

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NDVI750())
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
        product = product_generator.apply(sensor_bands_info(), macro_raster(), metadata=True)
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
        self.assertTrue((raster.band('red') == 138.0).all())
        self.assertTrue((raster.band('green') == 120.0).all())
        self.assertTrue((raster.band('blue') == 106.0).all())

    def test_match_bands_for_legend(self):
        product_generator = LandCoverIndex()
        expected_bands_mapping = {
            'red': ['HC_830'],
            'green': ['HC_690'],
            'blue': ['HC_550']
        }
        bands_mapping = product_generator.match_bands_for_legend(sensor_bands_info(), macro_raster().band_names)
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
        self.assertTrue((raster.band('red') == 100.0).all())
        self.assertTrue((raster.band('green') == 102.0).all())
        self.assertTrue((raster.band('blue') == 104.0).all())

    def test_match_bands_for_legend(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        expected_bands_mapping = {
            'red': ['red_enhanced'],
            'green': ['green_enhanced'],
            'blue': ['blue_enhanced']
        }
        bands_mapping = product_generator.match_bands_for_legend(sensor_bands_info(), macro_raster().band_names)
        self.assertCountEqual(bands_mapping, expected_bands_mapping)

    def test_product_fail_for_macro_raster(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        mr = macro_raster()
        with self.assertRaises(ProductError) as ex:
            product_generator.apply(sensor_bands_info(), mr)
        print(str(ex.exception))
        self.assertTrue('blue_enhanced' in str(ex.exception))
        self.assertTrue('red_enhanced' in str(ex.exception))
        self.assertTrue('green_enhanced' in str(ex.exception))


"""
import telluric as tl
from telluric.georaster.util.products import ProductFactory, BandsMapping
from telluric.georaster.util.product_view import ProductViewsFactory
# satellites util is the new repository with our bands settings
from satellites_utils import newsat_bands

# instanciate a prodacts factury for the selected supplier
products_factory = ProductFactory(newsat_bands)  # type: ProductFuctory

# open a multiband raster
raster = tl.GeoRaster2.open('test_raster.tif')

# retrieve all products matching this raster
products_list = products_factory.get_matchings(raster.band_names)

# generate a product
product_generator = products_factory.get_object('NDVI')  # type: ProductGenerator
product = product_generator.apply(raster)  # type: GeoRaster2

# generate a product view for the requested product using the product default view
renderer = ProductViewsFactory.get_object(product_generator.default_view)  # type: ProductView
product_view = renderer.apply(product)  # type: GeoRaster2

# generating a single bands product for band red and product view using jet colormap
single_red_generator = products_factory.get_object('singleband', 'red')  # type: ProductGenerator
red_band = single_red_generator.apply(raster)  # type: GeoRaster2
red_jet = ProductViewsFactory.get_object('cm-jet').apply(red_value)  # type: GeoRaster2

# building a band: when the user does not have the band to bandlength mapping
# the user is require to pass the list of bands in the right order and we generate the mapping based on our internal mapping

bands_mapping = BandsMapping(['red', 'green', 'blue']).generate()  # type Dict
# {
#     0: {'min': 590, 'mean': 640, 'max': 690},
#     1: {'min': 515, 'mean': 547.5, 'max': 580},
#     2: {'min': 400, 'mean': 455, 'max': 510},
# }

products_factory = ProductFactory(bands_mapping)  # type: ProductFuctory
# from here it continues the same

## MIXIN

# get all products applicable to raster
raster.get_products(bands_mapping)  # type list(str)

# apply product on raster
raster.apply_product(bands_mapping, 'NDVI')  # type: GeoRaster2

# apply product and visualization
raster.visualize_product(bands_mapping, 'NDVI', 'cm-jet')  # type: GeoRaster2

# apply product and visualization using default visualization
raster.visualize_product(bands_mapping, 'NDVI')  # type: GeoRaster2

# all above can return product metadata
raster.apply_product(bands_mapping, 'NDVI', metadata=True)  # type: tuple(GeoRaster2, dict)
raster.visualize_product(bands_mapping, 'NDVI', 'cm-jet', metadata=True)  # type: tuple(GeoRaster2, dict)
raster.visualize_product(bands_mapping, 'NDVI', metadata=True)  # type: tuple(GeoRaster2, dict)
"""
