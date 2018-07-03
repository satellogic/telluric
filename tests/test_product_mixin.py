import pytest
import numpy as np
from telluric import GeoRaster2

from common_for_tests import (
    multi_raster_8b, sensor_bands_info,
    hyper_raster, hyper_raster_with_no_data,
    multi_raster_with_no_data, multi_raster_16b,
    make_test_raster, multi_values_8b,
)

from telluric.products import ProductsFactory


def test_multispectral_raster_gets_available_products():
    products = multi_raster_8b().get_products(sensor_bands_info())
    expected_products = ProductsFactory.get_matchings(multi_raster_8b().band_names, sensor_bands_info())
    assert products == expected_products


def test_hyperspectral_raster_gets_available_products():
    products = hyper_raster().get_products(sensor_bands_info())
    expected_products = ProductsFactory.get_matchings(hyper_raster().band_names, sensor_bands_info())
    assert products == expected_products


@pytest.mark.parametrize('raster', [multi_raster_16b(), multi_raster_with_no_data(),
                                    hyper_raster(), hyper_raster_with_no_data()])
def test_apply_product(raster):
    for product_name in raster.get_products(sensor_bands_info()):
        if product_name == 'Custom':
            return
        if product_name == 'SingleBand':
            product = raster.apply(sensor_bands_info(), product_name, band=raster.band_names[0])
        else:
            product = raster.apply(sensor_bands_info(), product_name)

        assert isinstance(product, GeoRaster2)


@pytest.mark.parametrize("raster", [multi_raster_16b(),
                                    multi_raster_8b(),
                                    hyper_raster(),
                                    hyper_raster_with_no_data(),
                                    multi_raster_with_no_data()])
def test_it_visualized_to_colormap(raster):
    result = raster.apply(sensor_bands_info(), 'ndvi').visualize('cm-jet', vmax=1, vmin=-1)
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']


@pytest.mark.parametrize("raster", [multi_raster_16b(),
                                    multi_raster_8b(),
                                    hyper_raster(),
                                    hyper_raster_with_no_data(),
                                    multi_raster_with_no_data()])
def test_it_visualized_to_default(raster):
    product = raster.apply(sensor_bands_info(), 'ndvi', metadata=True)
    result = product.raster.visualize(product.default_view, vmax=product.max, vmin=product.min)
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']


@pytest.mark.parametrize("raster", [multi_raster_16b(),
                                    multi_raster_8b(),
                                    hyper_raster(),
                                    hyper_raster_with_no_data(),
                                    multi_raster_with_no_data()])
def test_it_visualized_to_rgb(raster):
    result = raster.apply(sensor_bands_info(), 'TrueColor').visualize('TrueColor')
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']


@pytest.mark.parametrize("raster", [multi_raster_16b(),
                                    multi_raster_8b(),
                                    multi_raster_with_no_data()])
def test_it_visualized_multispectral_to_rgb_with_no_product(raster):
    result = raster.visualize('TrueColor')
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']


def test_custom_product_preserve_mask():

    def foo(red, green):
        arr1 = red + 1
        arr2 = green + 2
        arr = np.stack([arr1[0], arr2[0]])
        return arr

    raster = multi_raster_with_no_data()
    output = raster.apply(sensor_bands_info(), 'custom', function=foo, required_bands=['red', 'green'],
                          output_bands=['red', 'green'])
    expected_mask = raster.bands_data('red').mask | raster.bands_data('green').mask
    assert(output.shape[1:] == raster.shape[1:])
    assert(output.band_names == ['red', 'green'])
    assert((output.bands_data('red') == multi_values_8b['red'] + 1).all())
    assert((output.bands_data('green') == multi_values_8b['green'] + 2).all())
    assert((output.image.mask == expected_mask).all())


def test_custom_tow_bands_plus_1_and_2_with_bands_mapping():

    def foo(red, green):
        arr1 = red + 1
        arr2 = green + 2
        arr = np.stack([arr1[0], arr2[0]])

        return arr

    bands_mapping = {'red': ['b1', 'b2'], 'green': ['b4', 'b5', 'b6']}
    raster = make_test_raster(value=66, band_names=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
    product = raster.apply(sensor_bands_info(), 'custom', function=foo, required_bands=['red', 'green'],
                           output_bands=['red', 'green'], bands_mapping=bands_mapping, metadata=True)
    output = product.raster
    assert output.shape[1:] == raster.shape[1:]
    assert output.band_names == ['red', 'green']
    assert (output.bands_data('red') == 67).all()
    assert (output.bands_data('green') == 68).all()
    assert product.bands_mapping, bands_mapping
