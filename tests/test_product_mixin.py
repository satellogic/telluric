import pytest
from telluric import GeoRaster2

from common_for_tests import (
    micro_raster_8b, sensor_bands_info,
    macro_raster, macro_raster_with_no_data,
    micro_raster_with_no_data, micro_raster_16b
)

from telluric.products import ProductsFactory


def test_micro_raster_gets_available_products():
    products = micro_raster_8b().get_products(sensor_bands_info())
    expected_products = ProductsFactory.get_matchings(micro_raster_8b().band_names, sensor_bands_info())
    assert products == expected_products


def test_macro_raster_gets_available_products():
    products = macro_raster().get_products(sensor_bands_info())
    expected_products = ProductsFactory.get_matchings(macro_raster().band_names, sensor_bands_info())
    assert products == expected_products


def test_micro_apply_product():
    raster = micro_raster_8b()
    for product_name in raster.get_products(sensor_bands_info()):
        if product_name == 'SingleBand':
            product = raster.apply(sensor_bands_info(), product_name, band=raster.band_names[0])
        else:
            product = raster.apply(sensor_bands_info(), product_name)

        assert isinstance(product, GeoRaster2)


def test_macro_apply_product():
    raster = macro_raster()
    for product_name in raster.get_products(sensor_bands_info()):
        if product_name == 'SingleBand':
            product = raster.apply(sensor_bands_info(), product_name, band=raster.band_names[0])
        else:
            product = raster.apply(sensor_bands_info(), product_name)

        assert isinstance(product, GeoRaster2)


@pytest.mark.parametrize("raster", [micro_raster_16b(),
                                    micro_raster_8b(),
                                    macro_raster(),
                                    macro_raster_with_no_data(),
                                    micro_raster_with_no_data()])
def test_it_visualized_to_colormap(raster):
    result = raster.apply(sensor_bands_info(), 'ndvi').visualize('cm-jet')
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']


@pytest.mark.parametrize("raster", [micro_raster_16b(),
                                    micro_raster_8b(),
                                    macro_raster(),
                                    macro_raster_with_no_data(),
                                    micro_raster_with_no_data()])
def test_it_visualized_to_default(raster):
    product = raster.apply(sensor_bands_info(), 'ndvi', metadata=True)
    result = product.raster.visualize(product.default_view, vmax=product.max, vmin=product.min)
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']

@pytest.mark.parametrize("raster", [micro_raster_16b(),
                                    micro_raster_8b(),
                                    macro_raster(),
                                    macro_raster_with_no_data(),
                                    micro_raster_with_no_data()])
def test_it_visualized_to_rgb_(raster):
    result = raster.apply(sensor_bands_info(), 'TrueColor').visualize('TrueColor')
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']

@pytest.mark.parametrize("raster", [micro_raster_16b(),
                                    micro_raster_8b(),
                                    micro_raster_with_no_data()])
def test_it_visualized_micro_to_rgb_with_no_product(raster):
    result = raster.visualize('TrueColor')
    assert isinstance(result, GeoRaster2)
    assert result.band_names == ['red', 'green', 'blue']
