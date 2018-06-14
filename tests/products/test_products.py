import unittest
import numpy as np

from affine import Affine
import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS

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


micro_values = {
    'green': 55,
    'red': 77,
    'nir': 100,
    'blue': 118
}

expected_nir = ['HyperCube_770_nm', 'HyperCube_827_nm']
expected_red = ['HyperCube_670_nm', 'HyperCube_608_nm']
expected_green = ['HyperCube_570_nm', 'HyperCube_550_nm', 'HyperCube_530_nm']
expected_blue = ['HyperCube_450_nm', 'HyperCube_502_nm']

# calculated micro values
nir_value = 135    # nir values 134 136 avg 135
red_value = 113    # red values 112 114 avg 113
green_value = 116  # green values 104 106 138 avg 105
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
    source_raster.image.data[0, 1, 2] = source_raster.nodata
    source_raster.image.data[1, 2, 3] = source_raster.nodata
    source_raster.image.data[2, 0, 0] = source_raster.nodata
    source_raster.image.data[3, 1, 3] = source_raster.nodata
    return source_raster


macro_wavelengths = (450, 502, 530, 570, 582, 595, 608, 670, 680,
                     690, 700, 710, 720, 730, 740, 750, 760, 770, 827, 550)

macro_bands = ["HyperCube_%i_nm" % wl for wl in macro_wavelengths]


def macro_raster():
    source_raster = make_test_raster(142, macro_bands, dtype=np.uint8)
    for i, band in enumerate(macro_bands):
        source_raster.image.data[i, :, :] = 2 * i + 100
    return source_raster


def macro_raster_with_no_data():
    source_raster = macro_raster()
    source_raster.image.data[:, 1, 2] = source_raster.nodata
    source_raster.image.data[2, 2, 3] = source_raster.nodata
    source_raster.image.data[5, 0, 0] = source_raster.nodata
    source_raster.image.data[14, 1, 3] = source_raster.nodata
    source_raster.image.data[15, 1, 3] = source_raster.nodata
    return source_raster


class TestProductsFuctory(unittest.TestCase):

    def test_it_retrieves_the_right_products(self):
        self.assertIsInstance(ProductsFactory.get_object('NDVI'), NDVI)
        self.assertIsInstance(ProductsFactory.get_object('ENDVI'), ENDVI)
        self.assertIsInstance(ProductsFactory.get_object('EVI2'), EVI2)

    def test_it_is_case_insensitive(self):
        self.assertIsInstance(ProductsFactory.get_object('ndvi'), NDVI)

    def test_it_is_raises_key_error_whene_product_not_exists(self):
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
        arr = np.ma.array(np.zeros(raster.shape, dtype=np.uint16))
        bands = {band_name: raster.band(band_name) for band_name in raster.band_names}
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, bands)
        self.assertTrue((output_arr == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_zero_uint8(self):
        raster = make_test_raster(42, ['nir', 'red'])
        arr = np.ma.array(np.zeros(raster.shape, dtype=np.uint8))
        bands = {band_name: raster.band(band_name) for band_name in raster.band_names}
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, bands)
        self.assertTrue((output_arr == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_zero_float32(self):
        raster = make_test_raster(0.42, ['nir', 'red'], dtype=np.float32)
        arr = np.ma.array(np.zeros(raster.shape, dtype=np.float32))
        bands = {band_name: raster.band(band_name) for band_name in raster.band_names}
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, bands)
        self.assertTrue((output_arr == 0).all())
        self.assertFalse(output_arr.mask.any())

    def test_setting_minimal_value_for_nodata_uint16(self):
        raster = make_test_raster(42, ['nir', 'red'], dtype=np.uint16)
        array = raster.image.data
        mask = raster.image.mask
        mask[:, :, :] = True
        raster = raster.copy_with(image=np.ma.array(data=array, mask=mask))
        arr = np.ma.array(np.zeros(raster.shape))
        bands = {band_name: raster.band(band_name) for band_name in raster.band_names}
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, bands)
        self.assertTrue(output_arr.mask.all())

    def test_setting_minimal_value_for_nodata_float32(self):
        raster = make_test_raster(0.42, ['nir', 'red'], dtype=np.float32)
        array = raster.image.data
        mask = raster.image.mask
        mask[:, :, :] = True
        raster = raster.copy_with(image=np.ma.array(data=array, mask=mask))
        arr = np.ma.array(np.zeros(raster.shape))
        # bands = {band_name: raster.band(band_name) for band_name in raster.band_names}
        output_arr = NDVI()._force_nodata_where_it_was_in_original_raster(arr, raster, raster.band_names)
        self.assertTrue(output_arr.mask.all())


class TestBandsMatching(unittest.TestCase):

    def test_it_matches_all_bands(self):
        self.assertTrue(set(ProductsFactory.get_matchings(
            ['blue', 'nir', 'red'])).issuperset(['NDVI', 'EVI2']))
        self.assertTrue(set(ProductsFactory.get_matchings(['nir', 'red'])).issuperset(['NDVI', 'EVI2']))
        self.assertNotIn('EXR', ProductsFactory.get_matchings(['nir', 'red']))
        self.assertEqual(set(ProductsFactory.get_matchings(['red'])), {'SingleBand'})


class TestNDVI(unittest.TestCase):

    def test_ndvi(self):
        product = NDVI().apply(micro_raster())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        expected_value = (micro_values['nir'] - micro_values['red']) / (
            micro_values['nir'] + micro_values['red']
        )
        self.assertTrue((raster.image.data == expected_value).all())

    def test_NDVI_product(self):
        product_generator = NDVI()
        product = product_generator.apply(micro_raster())
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
        product = NDVI().apply(micro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertFalse(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertTrue(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_NDVI_product_for_macro(self):
        product_generator = NDVI()
        product = product_generator.apply(macro_raster())
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
        self.assertCountEqual(product.used_bands(), expected_nir + expected_red)

    def test_for_macro_with_no_data(self):
        product = NDVI().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_NDVI_product_fail_for_macro_with_missing_restricted_bands(self):
        product_generator = NDVI()
        mr = macro_raster()
        restricted_bands = mr.band_names + ['b659']
        with self.assertRaises(KeyError) as ex:
            product_generator.apply(mr, bands_restriction=restricted_bands)
        self.assertEqual(str(ex.exception), "'raster lacks restricted bands: b659'")

    def test_NDVI_product_for_macro_with_all_restricted_bands(self):
        product_generator = NDVI()
        mr = macro_raster()
        restricted_bands = mr.band_names
        product = product_generator.apply(mr, bands_restriction=restricted_bands)
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, NDVI.output_bands)
        self.assertEqual(raster.dtype, NDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)


class TestEVI2(unittest.TestCase):

    def test_evi2(self):
        product = EVI2().apply(micro_raster())
        raster = product.raster
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
        product = product_generator.apply(micro_raster())
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
        product = EVI2().apply(micro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertFalse(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertTrue(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_EVI2_product_for_macro(self):
        product_generator = EVI2()
        product = product_generator.apply(macro_raster())
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
        product = EVI2().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EVI2.output_bands)
        self.assertEqual(raster.dtype, EVI2.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestENDVI(unittest.TestCase):

    def test_endvi(self):
        product = ENDVI().apply(micro_raster())
        raster = product.raster
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
        product = product_generator.apply(micro_raster())
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
        product = ENDVI().apply(micro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertTrue(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 1, 3] == raster.nodata)

    def test_ENDVI_product_for_macro(self):
        product_generator = ENDVI()
        product = product_generator.apply(macro_raster())
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
        product = ENDVI().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, ENDVI.output_bands)
        self.assertEqual(raster.dtype, ENDVI.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestExcessIndices(unittest.TestCase):

    def test_exg(self):
        product = EXG().apply(micro_raster())
        raster = product.raster
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
        product = product_generator.apply(micro_raster())
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
        product = EXG().apply(micro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 1, 3] == raster.nodata)

    def test_EXG_product_for_macro(self):
        product_generator = EXG()
        product = product_generator.apply(macro_raster())
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
        product = EXG().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXG.output_bands)
        self.assertEqual(raster.dtype, EXG.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_exr(self):
        product = EXR().apply(micro_raster())
        raster = product.raster
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
        product = product_generator.apply(micro_raster())
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
        product = product_generator.apply(macro_raster())
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
        product = EXR().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXR.output_bands)
        self.assertEqual(raster.dtype, EXR.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_exb(self):
        product = EXB().apply(micro_raster())
        raster = product.raster
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
        product = product_generator.apply(micro_raster())
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
        product = product_generator.apply(macro_raster())
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
        product = EXB().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, EXB.output_bands)
        self.assertEqual(raster.dtype, EXB.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestTrueColor(unittest.TestCase):

    def test_true_color(self):
        product = TrueColor().apply(micro_raster())
        raster = product.raster
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.num_bands, 3)
        # TODO what should I do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        for band in ['red', 'green', 'blue']:
            self.assertTrue((raster.band(band) == micro_raster()[band]).all())

    def test_TrueColor_product(self):
        product_generator = TrueColor()
        product = product_generator.apply(micro_raster())
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
        product = TrueColor().apply(micro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        # TODO what to do here
        # self.assertEqual(raster.dtype, TrueColor.type)
        self.assertEqual(raster.height, micro_raster().height)
        self.assertEqual(raster.width, micro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 1, 3] == raster.nodata)

    def test_TrueColor_product_for_macro(self):
        product_generator = TrueColor()
        product = product_generator.apply(macro_raster())
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
        product = TrueColor().apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.band_names, TrueColor.output_bands)
        self.assertEqual(raster.dtype, macro_raster().dtype)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)

    def test_fits_raster_bands_for_entire_macro_band(self):
        true_color = ProductsFactory.get_object('truecolor')
        fits = true_color.fits_raster_bands(macro_bands)
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_RGB(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['red, green, blue'])
        self.assertEqual(fits, False)

    def test_fits_raster_bands_false_for_part_of_the_macro_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(macro_bands[15:])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(macro_bands[:6])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(macro_bands[7:15])
        self.assertEqual(fits, False)

    def test_fits_raster_bands_true_for_one_band_per_range(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['b457', 'b550', 'b620'])
        self.assertEqual(fits, True)

    def test_fits_raster_bands_true_niglecting_out_of_range_bands(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['b444', 'b550', 'b620', 'b300'])
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(['b444', 'b550', 'b620', 'b585'])
        self.assertEqual(fits, True)
        fits = hs_true_color.fits_raster_bands(['b444', 'b550', 'b620', 'b700'])
        self.assertEqual(fits, True)

    def test_fits_raster_bands_false_for_bands_in_range_hols_and_out_of_renge(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        fits = hs_true_color.fits_raster_bands(['b444', 'b550', 'b585'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['b444', 'b512', 'b590'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['b444', 'b550', 'b700'])
        self.assertEqual(fits, False)
        fits = hs_true_color.fits_raster_bands(['b340', 'b550', 'b620'])
        self.assertEqual(fits, False)

    def test_hs_true_color(self):
        hs_true_color = ProductsFactory.get_object('truecolor')
        product = hs_true_color.apply(macro_raster())
        raster = product.raster
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
        product = hs_true_color.apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.band_names, hs_true_color.output_bands)
        self.assertEqual(raster.num_bands, 3)
        self.assertEqual(raster.dtype, hs_true_color.type)
        self.assertEqual(raster.height, macro_raster_with_no_data().height)
        self.assertEqual(raster.width, macro_raster_with_no_data().width)
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[1, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[2, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)
        self.assertFalse(raster.image.data[1, 1, 3] == raster.nodata)
        self.assertFalse(raster.image.data[2, 1, 3] == raster.nodata)
        self.assertFalse(raster.image.data[1, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[2, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[2, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)

    def test_HSTrueColor_product(self):
        product_generator = ProductsFactory.get_object('truecolor')
        product = product_generator.apply(macro_raster())
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        expected_bands_mapping = {'blue': ['b444', 'b456', 'b468', 'b481', 'b494', 'b507'],
                                  'green': ['b520', 'b532', 'b545', 'b558', 'b571'],
                                  'red': ['b648', 'b661', 'b673', 'b686', 'b597', 'b609', 'b622', 'b635']}

        self.assertCountEqual(product.bands_mapping, expected_bands_mapping)


class TestSingleBand(unittest.TestCase):

    def test_single_band(self):
        raster = micro_raster()
        product = ProductsFactory.get_object('SingleBand', 'blue').apply(raster)
        self.assertEqual(product.raster, raster.limit_to_bands(['blue']))

    def test_object_vs_class(self):
        obj = ProductsFactory.get_object('SingleBand', 'blue')
        cls = ProductsFactory.get_class('SingleBand')
        self.assertEqual(type(obj), cls)

    def test_SingleBand_product(self):
        product_generator = ProductsFactory.get_object('SingleBand', 'blue')
        product = product_generator.apply(micro_raster())
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
        product_generator = ProductsFactory.get_object('SingleBand', 'HyperCube_450_nm')
        product = product_generator.apply(macro_raster())
        self.assertEqual(product.output_bands, ['HyperCube_450_nm'])


class BaseTestMacroProduct(unittest.TestCase):

    def do_test_product_for_macro(self, product_generator, expected_bands_mapping,
                                  expected_bands_names, expected_value):
        product = product_generator.apply(macro_raster())
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping, expected_bands_mapping)
        self.assertTrue((product.raster.image.data == expected_value).all())
        self.assertCountEqual(product.used_bands(), expected_bands_names)
        return product

    def do_test_for_macro_with_no_data(self, product_generator):
        product = product_generator.apply(macro_raster_with_no_data())
        raster = product.raster
        self.assertEqual(raster.num_bands, 1)
        self.assertEqual(raster.band_names, product_generator.output_bands)
        self.assertEqual(raster.dtype, product_generator.type)
        self.assertEqual(raster.height, macro_raster().height)
        self.assertEqual(raster.width, macro_raster().width)
        return product


class TestPRI(BaseTestMacroProduct):

    def test_PRI_product_for_macro(self):
        expected_bands_mapping = {'R530': ['HyperCube_530_nm'], 'R570': ['HyperCube_570_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (106.0 - 104.0) / (106.0 + 104.0)
        self.do_test_product_for_macro(PRI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(PRI())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestNDVI827(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R827': ['HyperCube_827_nm'], 'R690': ['HyperCube_690_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (136.0 - 118.0) / (136.0 + 118.0)
        self.do_test_product_for_macro(NDVI827(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NDVI827())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestNRI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R570': ['HyperCube_570_nm'], 'R670': ['HyperCube_670_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (106.0 - 114.0) / (106.0 + 114.0)
        self.do_test_product_for_macro(NRI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NRI())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestGNDVI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R750': ['HyperCube_750_nm'], 'R550': ['HyperCube_550_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (130.0 - 138.0) / (130.0 + 138.0)
        self.do_test_product_for_macro(GNDVI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(GNDVI())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 1, 3] == raster.nodata)


class TestCCI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R530': ['HyperCube_530_nm'], 'R670': ['HyperCube_670_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (104.0 - 114.0) / (104.0 + 114.0)
        self.do_test_product_for_macro(CCI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(CCI())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertTrue(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestNPCI(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R582': ['HyperCube_582_nm'], 'R450': ['HyperCube_450_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (108.0 - 100.0) / (108.0 + 100.0)
        self.do_test_product_for_macro(NPCI(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NPCI())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestPPR(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R550': ['HyperCube_550_nm'], 'R450': ['HyperCube_450_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (138.0 - 100.0) / (138.0 + 100.0)
        self.do_test_product_for_macro(PPR(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(PPR())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertFalse(raster.image.data[0, 1, 3] == raster.nodata)


class TestNDVI750(BaseTestMacroProduct):

    def test_product_for_macro(self):
        expected_bands_mapping = {'R750': ['HyperCube_750_nm'], 'R700': ['HyperCube_700_nm']}
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        expected_value = (130.0 - 120.0) / (130.0 + 120.0)
        self.do_test_product_for_macro(NDVI750(), expected_bands_mapping,
                                       expected_bands_names, expected_value)

    def test_for_macro_with_no_data(self):
        product = self.do_test_for_macro_with_no_data(NDVI750())
        raster = product.raster
        self.assertTrue(raster.image.data[0, 1, 2] == raster.nodata)
        self.assertFalse(raster.image.data[0, 2, 3] == raster.nodata)
        self.assertFalse(raster.image.data[0, 0, 0] == raster.nodata)
        self.assertTrue(raster.image.data[0, 1, 3] == raster.nodata)


class TestLandCoverIndex(unittest.TestCase):

    def test_product_for_macro(self):
        product_generator = LandCoverIndex()
        expected_bands_mapping = {
            'red': ['HyperCube_827_nm'],
            'green': ['HyperCube_690_nm'],
            'blue': ['HyperCube_550_nm']
        }
        expected_bands_names = [v[0] for v in expected_bands_mapping.values()]
        product = product_generator.apply(macro_raster())
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping, expected_bands_mapping)
        self.assertCountEqual(product.used_bands(), expected_bands_names)
        raster = product.raster
        self.assertTrue((raster.band('red') == 136.0).all())
        self.assertTrue((raster.band('green') == 118.0).all())
        self.assertTrue((raster.band('blue') == 138.0).all())

    def test_match_bands_for_legend(self):
        product_generator = LandCoverIndex()
        expected_bands_mapping = {
            'red': ['HyperCube_827_nm'],
            'green': ['HyperCube_690_nm'],
            'blue': ['HyperCube_550_nm']
        }
        bands_mapping = product_generator.match_bands_for_legend(macro_raster().band_names)
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
        product = product_generator.apply(self.enhanced_raster())
        self.assertEqual(product.name, product_generator.name)
        self.assertEqual(product.display_name, product_generator.display_name)
        self.assertEqual(product.description, product_generator.description)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.min, product_generator.min)
        self.assertEqual(product.required_bands, product_generator.required_bands)
        self.assertEqual(product.output_bands, product_generator.output_bands)
        self.assertEqual(product.unit, product_generator.unit)
        self.assertCountEqual(product.bands_mapping, expected_bands_mapping)
        self.assertCountEqual(product.used_bands(), expected_bands_names)
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
        bands_mapping = product_generator.match_bands_for_legend(macro_raster().band_names)
        self.assertCountEqual(bands_mapping, expected_bands_mapping)

    def test_product_fail_for_macro_raster(self):
        product_generator = ProductsFactory.get_object('rgbenhanced')
        mr = macro_raster()
        with self.assertRaises(KeyError) as ex:
            product_generator.apply(mr)
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
