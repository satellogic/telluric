from functools import reduce
from collections import OrderedDict
from math import floor, ceil
import numpy as np

from telluric.util.raster_utils import _join_masks_from_masked_array


from .base_renderer import BaseFactory
# from rastile.satellite import sat_utils
# from bands_definitions import bands_definition


def flatten_two_level_list(main_list):
    """Converts list of lists in to a single level list"""
    return [item for sublist in main_list for item in sublist]


class Product():
    """ this class is the result of applying a product on a raster"""

    def __init__(self, raster, bands_mapping, product_generator, **overrides):
        product_generator.update(overrides)
        self.raster = raster
        self.bands_mapping = bands_mapping
        self.name = product_generator['name']
        self.display_name = product_generator['display_name']
        self.description = product_generator['description']
        self.min = product_generator['min']
        self.max = product_generator['max']
        self.type = product_generator['type']
        self.required_bands = product_generator['required_bands']
        self.output_bands = product_generator['output_bands']
        self.unit = product_generator['unit']

    def used_bands(self):
        return flatten_two_level_list(self.bands_mapping.values())  # flatten bands list


class ProductsFactory(BaseFactory):
    __objects = None

    @classmethod
    def objects(cls):
        if not cls.__objects:
            subclasses = sorted(ProductGenerator.__subclasses__(), key=lambda p: (p._order, p.name.lower()))
            cls.__objects = OrderedDict()
            for p in subclasses:
                if p.dont_add_to_factory:
                    continue
                cls.__objects[p.name.lower()] = p
        return cls.__objects


class ProductGenerator:
    """ProductGenerator is an abstract class.

    To implement a product class you should inherit from it.
    You should implement _apply
    You should set the following attributes:
        name min max required_bands output_bands description default_view type display_name unit _order
    """

    should_convert_to_float = True

    # don't add this Product generator to ProductsFactory list
    dont_add_to_factory = False

    # dicts with the wavelengths for products band definitions
    wavelength_map = {
        'blue': {'min': 400, 'mean': 455, 'max': 510},
        'green': {'min': 515, 'mean': 547.5, 'max': 580},
        'pan': {'min': 400, 'mean': 575, 'max': 750},
        'red': {'min': 590, 'mean': 640, 'max': 690},
        'nir': {'min': 750, 'mean': 825, 'max': 900},
        # new micro def
        'R450': {'min': 443, 'mean': 450, 'max': 457},
        'R530': {'min': 521, 'mean': 530, 'max': 539},
        'R550': {'min': 540, 'mean': 550, 'max': 560},
        'R570': {'min': 560, 'mean': 570, 'max': 580},
        'R582': {'min': 572, 'mean': 582, 'max': 592},
        'R670': {'min': 657, 'mean': 670, 'max': 683},
        'R690': {'min': 676, 'mean': 690, 'max': 703},  # should be confirmed with Martin
        'R700': {'min': 686, 'mean': 700, 'max': 714},
        'R750': {'min': 735, 'mean': 750, 'max': 765},
        'R827': {'min': 809, 'mean': 827, 'max': 845},  # should be confirmed with Martin
    }

    @classmethod
    def to_dict(cls):
        attributes = "name min max required_bands output_bands description default_view type display_name unit".split()
        d = {}
        for attr in attributes:
            d[attr] = getattr(cls, attr)
        return d

    def _apply(self, **bands):
        raise NotImplementedError

    @classmethod
    def band_metadata(cls, band_name, metadata_source, meta, default=None):
        band_meta = metadata_source.get(band_name, None)
        if band_meta:
            if meta:
                return band_meta[meta]
            else:
                return band_meta
        else:
            return default

    @classmethod
    def max_wavelength(cls, scensor_bands_info, band_name, default=None):
        return cls.band_metadata(band_name, scensor_bands_info, 'max', default)

    @classmethod
    def min_wavelength(cls, scensor_bands_info, band_name, default=None):
        return cls.band_metadata(band_name, scensor_bands_info, 'min', default)

    @classmethod
    def min_product_wavelength(cls, band_name):
        return cls.band_metadata(band_name, cls.wavelength_map, 'min')

    @classmethod
    def max_product_wavelength(cls, band_name):
        return cls.band_metadata(band_name, cls.wavelength_map, 'max')

    @classmethod
    def match_bands(cls, scensor_bands_info, bands_list, dest_band):
        min_val = cls.min_product_wavelength(dest_band)
        max_val = cls.max_product_wavelength(dest_band)
        return [a for a in bands_list if (min_val <= ceil(cls.min_wavelength(scensor_bands_info, a, default=0)) and
                                          floor(cls.max_wavelength(scensor_bands_info, a, default=1000)) <= max_val)]

    @classmethod
    def fits_raster_bands(cls, scensor_bands_info, available_bands, bands_restriction=None, silent=True):
        bands_restriction = bands_restriction or available_bands
        matched_bands = cls.match_available_bands_to_required_bands(scensor_bands_info, bands_restriction)
        if matched_bands:
            do_we_have_elements_to_create_required_bands = all(
                [len(a) > 0 for a in matched_bands.values()])
            if do_we_have_elements_to_create_required_bands:
                used_restricted_bands = flatten_two_level_list(matched_bands.values())
                if set(available_bands).issuperset(used_restricted_bands):
                    return True
                if silent:
                    return False
                missing_restricted_bands = [a for a in used_restricted_bands if a not in available_bands]
                raise KeyError('raster lacks restricted bands: %s' % ','.join(missing_restricted_bands))

        if silent:
            return False
        zero_bands = [a for a, v in matched_bands.items() if len(v) == 0]
        raise KeyError('raster lacks bands for: %s' % ','.join(zero_bands))

    @classmethod
    def match_available_bands_to_required_bands(cls, scensor_bands_info, available_bands, required_bands=None):
        matched_bands = {}
        required_bands = required_bands or cls.required_bands
        bands = set(available_bands)
        for dest_band in required_bands:
            matched_bands[dest_band] = cls.match_bands(scensor_bands_info, bands, dest_band)
        return matched_bands

    @classmethod
    def match_bands_for_legend(cls, scensor_bands_info, available_bands):
        bands_mapping = cls.match_available_bands_to_required_bands(scensor_bands_info, available_bands)
        bands_mapping = cls.override_bands_mapping(bands_mapping)
        return bands_mapping

    def extract_bands(self, raster, bands_mapping):
        """return bands in required_bands if required_bands is empty it returns all bands"""
        bands = {dest_band: self._merge_bands(bands_mapping[dest_band], raster)
                 for dest_band in self.required_bands}
        if self.should_convert_to_float:
            bands = {band_name: band.astype(np.float32) for band_name, band in bands.items()}
        return bands

    def override_product_settings(self, product):
        """this function is used in subclasses to modify products settings"""
        return product

    @classmethod
    def override_bands_mapping(cls, bands_mapping):
        """this function is used in subclasses to modify bands_mapping"""
        return bands_mapping

    def apply(self, raster, scensor_bands_info, bands_restriction=None, **kwargs):
        """
        Apply product calculation on the raster

        :param raster: the data raster the product should apply on
        :param bands_restriction: not required list of bands required for the processing of the product,
            if one of this bands not exists in the raster the product should return `no data`
        :param kwargs: additional arguments
        """
        self.fits_raster_bands(raster.band_names, bands_restriction, silent=False)
        bands_mapping = self.match_available_bands_to_required_bands(scensor_bands_info, raster.band_names)
        bands = self.extract_bands(raster, bands_mapping)
        # to silence error on 0 division, which happens at nodata
        with np.errstate(divide='ignore', invalid='ignore'):
            array = self._apply(**bands)  # actual renderer calculation, implemented by derived class
        used_band_names = [item for sublist in bands_mapping.values() for item in sublist]
        array = self._force_nodata_where_it_was_in_original_raster(array, raster, used_band_names)

        #convert array to ma.array
        product_raster = raster.copy_with(image=array, band_names=self.output_bands)
        product = Product(raster=product_raster, bands_mapping=bands_mapping, product_generator=self.to_dict())
        product = self.override_product_settings(product)
        return product

    def _merge_bands(self, band_names, raster):
        bands = [raster.band(band_name) for band_name in band_names]
        mask = _join_masks_from_masked_array(bands)
        ma = np.ma.array(bands, mask=mask)
        combined_band = ma.mean(axis=0).filled(0).astype(raster.dtype)
        return combined_band

    def _force_nodata_where_it_was_in_original_raster(self, array, raster, band_names):
        # if np.issubdtype(array.dtype, float):
        #     array.data[array.mask] = 1e-10  # firs make all nodata minimal value 'epsilon'
        # else:
        #     array.data[array.mask] = 1  # firs make all nodata minimal value '1'
        no_data_mask = self._get_nodata_musk(raster, band_names)
        no_data_mask = np.logical_or(array.mask, no_data_mask)
        array[no_data_mask] = 0
        new_array = np.ma.array(array.data, mask=no_data_mask)
        return new_array

    def _get_nodata_musk(self, raster, band_names=None):

        band_names = band_names or raster.band_names
        bands = [raster.band(band_name).mask for band_name in band_names]
        mask = reduce(np.logical_or, bands)
        mask = np.stack([mask] * len(band_names))
        return mask


class NDVI(ProductGenerator):
    name = "NDVI"
    display_name = "NDVI"
    description = 'Normalized difference vegetation index'
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'red', 'nir'}
    output_bands = ['ndvi']
    unit = None
    _order = 4

    def _apply(cls, nir, red):
        # (NIR-RED)/(NIR+RED)
        array = np.divide(nir - red, nir + red)
        return array


class EVI2(ProductGenerator):
    name = "EVI2"
    display_name = "EVI2"
    description = 'Enhanced vegetation index 2'
    default_view = 'cm-RdYlGn'
    min = -1.05
    max = 2.5
    type = np.float32
    required_bands = {'red', 'nir'}
    output_bands = ['evi2']
    unit = None
    _order = 7

    def _apply(cls, nir, red):
        # 2.5*((NIR-RED)/(NIR+2.4*RED+1))
        array = 2.5 * np.divide(nir - red, nir + 2.4 * red + 1)
        return array


class ENDVI(ProductGenerator):
    name = "ENDVI"
    display_name = "ENDVI"
    description = 'Enhanced Normalized Difference Vegetation Index'
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'blue', 'green', 'nir'}
    output_bands = ['endvi']
    unit = None
    _order = 6

    def _apply(cls, nir, green, blue):
        # (NIR+GREEN-2*BLUE)/(NIR+GREEN+2*BLUE)
        array = np.divide(nir + green - 2 * blue, nir + green + 2 * blue)
        return array


class EXG(ProductGenerator):
    name = "ExG"
    display_name = "ExG"
    description = 'Excess Green Index'
    # default_view = 'cm-BrBG'
    default_view = 'cm-Greens'
    min = -1
    max = 2
    type = np.float32
    required_bands = {'blue', 'green', 'red'}
    output_bands = ['exg']
    unit = None
    _order = 7

    def _apply(cls, red, green, blue):
        # 2 * (Green / (Red + Green + Blue) – (Red / (Red + Green + Blue) – (Blue / (Red + Green + Blue)
        sum_all = red + green + blue
        r = np.divide(red, sum_all)
        g = np.divide(green, sum_all)
        b = np.divide(blue, sum_all)
        array = 2 * g - r - b
        return array


class EXR(ProductGenerator):
    name = "ExR"
    display_name = "ExR"
    description = 'Excess Red Index'
    # default_view = 'cm-RdGy_r'
    default_view = 'cm-Reds'
    min = -1
    max = 1.4
    type = np.float32
    required_bands = {'blue', 'green', 'red'}
    output_bands = ['exr']
    unit = None
    _order = 7

    def _apply(cls, red, green, blue):
        # 1.4 * (Red / (Red + Green + Blue) – (Green / (Red + Green + Blue)
        sum_all = red + green + blue
        r = np.divide(red, sum_all)
        g = np.divide(green, sum_all)
        array = 1.4 * r - g
        return array


class EXB(ProductGenerator):
    name = "ExB"
    description = 'Excess Blue Index'
    display_name = "ExB"
    # default_view = 'cm-terrain_r'
    default_view = 'cm-Blues'
    min = -1
    max = 1.4
    type = np.float32
    required_bands = {'blue', 'green', 'red'}
    output_bands = ['exr']
    unit = None
    _order = 7

    def _apply(cls, red, green, blue):
        # 1.4 * (Blue / (Red + Green + Blue) – (Green / (Red + Green + Blue)
        sum_all = red + green + blue
        g = np.divide(green, sum_all)
        b = np.divide(blue, sum_all)
        array = 1.4 * b - g
        return array


class SingleBand(ProductGenerator):
    name = "SingleBand"
    display_name = "SingleBand"
    description = 'Single band'
    default_view = 'SingleBand'
    min = None
    max = None
    type = None
    required_bands = {}
    output_bands = []
    unit = 'DN'
    _order = 3

    def __init__(self, band):
        self.required_bands = {band}
        self.output_bands = [band]

    @classmethod
    def fits_raster_bands(cls, available_bands, silent=True):
        if len(available_bands) >= 1:
            return True
        if silent:
            return False
        raise KeyError('expected 1 band, got: %s' % available_bands)

    # @classmethod
    # def min_wavelength(cls, band_name, default=None):
    #     return cls.band_metadata(band_name, sat_utils.wavelength_map, 'min', default)

    # @classmethod
    # def min_product_wavelength(cls, band_name, **):
    #     return cls.min_wavelength(band_name)

    # @classmethod
    # def max_product_wavelength(cls, band_name):
    #     return cls.max_wavelength(band_name)

    @classmethod
    def match_bands(cls, _, bands_list, dest_band):
        return [dest_band] if dest_band in bands_list else []

    def _apply(self, band, **kwargs):
        array = band
        return array

    # def apply(self, raster, **kwargs):
    #     bands_mapping = self.match_available_bands_to_required_bands(raster.band_names,
    #                                                                  required_bands=self.required_bands)
    #     product_raster = raster.limit_to_bands(self.output_bands)
    #     return Product(raster=product_raster, bands_mapping=bands_mapping, product_generator=self.to_dict(),
    #                    required_bands=self.required_bands, output_bands=self.output_bands)


class TrueColor(ProductGenerator):
    name = "TrueColor"
    display_name = "TrueColor"
    description = 'RGB'
    default_view = 'TrueColor'
    min = 0
    max = 255
    type = np.uint8
    required_bands = {'red', 'green', 'blue'}
    output_bands = ['red', 'green', 'blue']
    unit = 'DN'
    should_convert_to_float = False
    _order = 1

    def _apply(self, red, green, blue, **kwargs):
        array = np.stack((red[0], green[0], blue[0]))
        return array


class RGBEnhanced(ProductGenerator):
    name = "RGBEnhanced"
    display_name = "RGB Enhanced"
    description = 'Color enhanced version of RGB'
    default_view = 'TrueColor'
    min = 0
    max = 255
    type = np.uint8
    required_bands = {'red_enhanced', 'green_enhanced', 'blue_enhanced'}
    output_bands = ['red', 'green', 'blue']
    unit = 'DN'
    should_convert_to_float = False
    _order = 1

    @classmethod
    def match_bands(cls, _, bands_list, dest_band):
        return [dest_band] if dest_band in bands_list else []

    def _apply(self, red_enhanced, green_enhanced, blue_enhanced, **kwargs):
        array = np.stack((red_enhanced[0], green_enhanced[0], blue_enhanced[0]))
        return array

    @classmethod
    def override_bands_mapping(cls, bands_mapping):
        new_bands_mapping = {
            'blue': bands_mapping['blue_enhanced'],
            'green': bands_mapping['green_enhanced'],
            'red': bands_mapping['red_enhanced']
        }
        return new_bands_mapping

    def override_product_settings(self, product):
        product.bands_mapping = self.override_bands_mapping(product.bands_mapping)
        return product


####
# Land Cover Index
# 550 - blue
# 690 - green
# 827 - red
####
class LandCoverIndex(ProductGenerator):
    name = "LandCoverIndex"
    display_name = "SCDI: Satellogic Crop Discriminant Index"
    description = 'Satellogic Crop Discriminant Index: False Color Index to maximize crop separation'
    default_view = 'TrueColor'
    min = 0
    max = 255
    type = np.uint8
    required_bands = {'R550', 'R690', 'R827'}
    output_bands = ['red', 'green', 'blue']
    unit = 'DN'
    should_convert_to_float = False
    _order = 5

    def _apply(self, R550, R690, R827, **kwargs):
        red = R827
        green = R690
        blue = R550
        array = np.stack((red[0], green[0], blue[0]))
        return array

    @classmethod
    def override_bands_mapping(cls, bands_mapping):
        new_bands_mapping = {
            'blue': bands_mapping['R550'],
            'green': bands_mapping['R690'],
            'red': bands_mapping['R827']
        }
        return new_bands_mapping

    def override_product_settings(self, product):
        product.bands_mapping = self.override_bands_mapping(product.bands_mapping)
        return product


class FalseColor(ProductGenerator):
    name = "FalseColor"
    display_name = "FalseColor"
    description = 'RGB interpretation of a composition of bands'
    default_view = 'TrueColor'
    min = 0
    max = 255
    type = np.uint8
    required_bands = {}
    output_bands = ['red', 'green', 'blue']
    unit = 'DN'
    should_convert_to_float = False
    _order = 5
    dont_add_to_factory = True

    def __init__(self, band_mapping):
        assert list(band_mapping.keys()).sort() == ['blue', 'green', 'red'].sort()
        self.required_bands = set(band_mapping.values())
        self.band_mapping = band_mapping

    def _apply(self, **bands):
        red = bands[self.band_mapping['red']]
        green = bands[self.band_mapping['green']]
        blue = bands[self.band_mapping['blue']]
        array = np.stack((red[0], green[0], blue[0]))
        return array


# PRI - done
# Photochemical Reflectance Index
# Radiation use efficiency
# (R570-R530)/ (R570+R530)
# The Photochemical Reflectance Index (PRI) is a surrogate for the radiation use efficiency of the vegetation.
# PRI measures the relative reflectance on either side of the green reflectance “hump” (550 nm),
# it also compares the reflectance in the blue (chlorophyll and carotenoids absorption)
# region of the spectrum with the reflectance in the red (chlorophyll absorption only) region.
# The Satellogic PRI is calculated as (R(560 - 580)-R(521 - 539))/(R(560 - 580)+R(521 - 539))

class PRI(ProductGenerator):
    name = "PRI"
    display_name = "Photochemical Reflectance Index"
    description = """
    The Photochemical Reflectance Index (PRI) is a surrogate for the radiation use efficiency of the vegetation.
    PRI measures the relative reflectance on either side of the green reflectance “hump” (550 nm),
    it also compares the reflectance in the blue (chlorophyll and carotenoids absorption)
    region of the spectrum with the reflectance in the red (chlorophyll absorption only) region.
    The Satellogic PRI is calculated as (R(560 - 580)-R(521 - 539))/(R(560 - 580)+R(521 - 539))
    """

    default_view = 'cm-jet'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R530', 'R570'}
    output_bands = ['pri']
    unit = None
    _order = 5

    def _apply(cls, R570, R530):
        # (R570-R530)/ (R570+R530)
        array = np.divide(R570 - R530, R570 + R530)
        return array


# NDVI827
# Normalized difference vegetation index
# Radiation interception, Leaf Area Index
# (R827-R690)/ (R827+R690)
# The Normalized Difference Vegetation Index (NDVI) it can be use as an estimator of the photosynthetic capacity of the
# vegetation, the leaf area index or the phenology of the vegetation. It is widely used as an estimator of the fraction
# of the photosynthetic active radiation intercepted by the vegetation.
# Valid range is from -1 (clouds, ice) to 1 (very active vegetation).

class NDVI827(ProductGenerator):
    name = "NDVI827"
    display_name = "Normalized difference vegetation index"
    description = """
    The Normalized Difference Vegetation Index (NDVI) it can be use as an estimator of the photosynthetic capacity of
    the vegetation, the leaf area index or the phenology of the vegetation.
    It is widely used as an estimator of the fraction of the photosynthetic active radiation intercepted by the
    vegetation.
    Valid range is from -1 (clouds, ice) to 1 (very active vegetation).
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R827', 'R690'}
    output_bands = ['ndvi827']
    unit = None
    _order = 5

    def _apply(cls, R827, R690):
        # (R827-R690)/ (R827+R690)
        array = np.divide(R827 - R690, R827 + R690)
        return array


# NRI
# Nitrogen reflectance index
# Nitrogen stress (fertilization management and decisions)
# (R570-R670)/ (R570+R670)
# The Nitrogen reflectance index (NRI) is directly related to N concentration in canopies completely covering the soil.
# This index could be used for nutritional stress detection and  fertilization management.
# The Satellogic NRI is calculated as (R(560 - 580)-R(657 - 683)/(R(560 - 580)+R(657 - 683))

class NRI(ProductGenerator):
    name = "NRI"
    display_name = "Nitrogen reflectance index"
    description = """
    The Nitrogen reflectance index (NRI) is directly related to N concentration in canopies
    completely covering the soil.
    This index could be used for nutritional stress detection and  fertilization management.
    The Satellogic NRI is calculated as (R(560 - 580)-R(657 - 683)/(R(560 - 580)+R(657 - 683))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R570', 'R670'}
    output_bands = ['nri']
    unit = None
    _order = 5

    def _apply(cls, R570, R670):
        # (R570-R670)/ (R570+R670)
        array = np.divide(R570 - R670, R570 + R670)
        return array


# GNDVI
# Green NDVI
# Photosynthesis, productivity
# (R750-R550)/ (R750+R550)
# The Green Normalized Difference Vegetation Index (GNDVI) is an estimator of chlorophyll content at the canopy level.
# The Satellogic GNDVI is calculated as (R(735 - 765)-R(540 - 560))/(R(735 - 765)+R(540 - 560))

class GNDVI(ProductGenerator):
    name = "GNDVI"
    display_name = "Green NDVI"
    description = """
    The Green Normalized Difference Vegetation Index (GNDVI) is an estimator of chlorophyll content at the canopy level.
    The Satellogic GNDVI is calculated as (R(735 - 765)-R(540 - 560))/(R(735 - 765)+R(540 - 560))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R750', 'R550'}
    output_bands = ['gndvi']
    unit = None
    _order = 5

    def _apply(cls, R750, R550):
        # (R750-R550)/ (R750+R550)
        array = np.divide(R750 - R550, R750 + R550)
        return array


# CCI
# Chlorophyll/Carotenoid Index
# Photosynthetic rates
# (R530-R670)/(R530+R670)
# The Chlorophyll/Carotenoid Index (CCI) is sensitive to seasonally changing chlorophyll/carotenoid pigment ratios and
# is a tracking tool for photosynthetic phenology in evergreen vegetation.
# The CCI reveals seasonally changing photosynthetic rates and can detect the onset of the growing season in evergreen
# vegetation.
# The Satellogic CCI is calculated as (R(521 - 539)-R(657-683))/(R(521 - 539)+R(657-683))

class CCI(ProductGenerator):
    name = "CCI"
    display_name = "Chlorophyll/Carotenoid Index"
    description = """
    The Chlorophyll/Carotenoid Index (CCI) is sensitive to seasonally changing chlorophyll/carotenoid pigment ratios and
    is a tracking tool for photosynthetic phenology in evergreen vegetation.
    The CCI reveals seasonally changing photosynthetic rates and can detect the onset of the growing season in evergreen
    vegetation.
    The Satellogic CCI is calculated as (R(521 - 539)-R(657-683))/(R(521 - 539)+R(657-683))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R530', 'R670'}
    output_bands = ['cci']
    unit = None
    _order = 5

    def _apply(cls, R530, R670):
        # (R530-R670)/(R530+R670)
        array = np.divide(R530 - R670, R530 + R670)
        return array


# NPCI
# Normalized pigment chlorophyll index
# Chlorophyll/pigments ratio, Nitrogen content
# (R582-R450)/(R582+R450)
# The Normalized pigment chlorophyll index (NPCI) is a surrogate for nitrogen content in the vegetation because
# it is sensitive to the relation between total photosynthetic pigments to chlorophyll.
# N limited plants develop greater concentration of carotenoids relative to chlorophyll,
# thus the NPCI can detect this ratio.
# The Satellogic NPCI is calculated as (R(572 - 592)-R(443 - 457))/(R(572 - 592)+R(443 - 457))

class NPCI(ProductGenerator):
    name = "NPCI"
    display_name = "Normalized pigment chlorophyll index"
    description = """
    The Normalized pigment chlorophyll index (NPCI) is a surrogate for nitrogen content in the vegetation because
    it is sensitive to the relation between total photosynthetic pigments to chlorophyll.
    N limited plants develop greater concentration of carotenoids relative to chlorophyll,
    thus the NPCI can detect this ratio.
    The Satellogic NPCI is calculated as (R(572 - 592)-R(443 - 457))/(R(572 - 592)+R(443 - 457))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R582', 'R450'}
    output_bands = ['npci']
    unit = None
    _order = 5

    def _apply(cls, R582, R450):
        # (R582-R450)/(R582+R450)
        array = np.divide(R582 - R450, R582 + R450)
        return array


# PPR
# Plant pigment ratio
# Wheat grain protein content
# (R550-R450)/(R550+R450)
# The Plant pigment ratio (PPR) is sensitive to leaf Chl concentration and leaf N concentration.
# PPR is a promising indicator to predict wheat grain protein for different genotypes.
# Under the experimental conditions, the PPR at 18 DAA and anthesis
# were the best predictors of grain protein in different types of weath.
# The Satellogic PPR index is calculated as (R(540 - 560)-R(443 - 457))/(R(540 - 560)+R(443 - 457))

class PPR(ProductGenerator):
    name = "PPR"
    display_name = "Plant pigment ratio"
    description = """
    The Plant pigment ratio (PPR) is sensitive to leaf Chl concentration and leaf N concentration.
    PPR is a promising indicator to predict wheat grain protein for different genotypes.
    Under the experimental conditions, the PPR at 18 DAA and anthesis
    were the best predictors of grain protein in different types of weath.
    The Satellogic PPR index is calculated as (R(540 - 560)-R(443 - 457))/(R(540 - 560)+R(443 - 457))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R550', 'R450'}
    output_bands = ['ppr']
    unit = None
    _order = 5

    def _apply(cls, R550, R450):
        # (R550-R450)/(R550+R450)
        array = np.divide(R550 - R450, R550 + R450)
        return array


# NDVI705
# Red Edge Normalized Vegetation Index
# (R750-R700)/(R750+R700)
# The Red Edge Normalized Vegetation Index (NDVI705) take advantage on the sensitivity of the vegetation red edge
# to small changes in canopy foliage content, gap fraction, and senescence.
# Applications include precision agriculture, forest monitoring, and vegetation stress detection.
# The values of this index range from -1 to 1.The Satellogic NDVI705 is calculated as:
# (R(735 - 765)-R(686 - 714))/(R(735 - 765)+R(686 - 714))

class NDVI750(ProductGenerator):
    name = "NDVI750"
    display_name = "Red Edge Normalized Vegetation Index"
    description = """
    The Red Edge Normalized Vegetation Index (NDVI705) take advantage on the sensitivity of the vegetation red edge
    to small changes in canopy foliage content, gap fraction, and senescence.
    Applications include precision agriculture, forest monitoring, and vegetation stress detection.
    The values of this index range from -1 to 1.The Satellogic NDVI705 is calculated as:
    (R(735 - 765)-R(686 - 714))/(R(735 - 765)+R(686 - 714))
    """
    default_view = 'cm-RdYlGn'
    min = -1
    max = 1
    type = np.float32
    required_bands = {'R750', 'R700'}
    output_bands = ['ndvi750']
    unit = None
    _order = 5

    def _apply(cls, R750, R700):
        # (R750-R700)/(R750+R700)
        array = np.divide(R750 - R700, R750 + R700)
        return array
