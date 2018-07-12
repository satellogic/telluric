"""Mixin to make products functionality first members in GeoRaster2."""
from telluric.products import ProductsFactory, ProductError
from telluric.product_view import ProductViewsFactory
from telluric.context import local_context


class ProductsMixin:
    def apply(self, product_name, sensor_bands_info=None, metadata=False, **kwargs):
        """Apply a product on a raster, from a pre defined list of products, such as NVDI.

        Parameters
        ----------
        product_name: str
            Name of requested product
        sensor_bands_info: dict
            Bands wave length definition including max and min wave per bands name, if it is not set as an argument
            it should be set in TelluricContext
        metadata: bool
            Optional, when True it returns a namedtuple withe resulting raster and additional product information
        kwargs: dict
            Additional arguments, required to generate the product like the band name for SingleBand product

        Returns
        --------
        GeoRaster2
        Product()
        """
        if sensor_bands_info is None:
            sensor_bands_info = local_context.get('sensor_bands_info')
        if sensor_bands_info is None:
            raise ProductError("sensor_bands_info must be supplied")

        generator = ProductsFactory.get_object(product_name, **kwargs)
        product = generator.apply(sensor_bands_info, self, metadata=metadata)
        return product

    def visualize(self, product_view_name, **kwargs):
        """Converts raster to RGB representation based on visualization name.

        Visualized a raster using a colormap, TrueColor or other.

        Parameters
        ----------
        product_view_name: str
            Name of requested visualization
        kwargs: dict
            Additional arguments if required

        Rerturns
        --------
        GeoRaster2
        """
        generator = ProductViewsFactory.get_object(product_view_name)
        view = generator.apply(self, **kwargs)
        return view

    def get_products(self, sensor_bands_info=None):
        """Return a list of product names applicable to the raster.

        Parameters
        ----------
        sensor_bands_info: dict
            Bands wave length definition including max and min wave per bands name, if it is not set as an argument
            it should be set in TelluricContext

        Rerturns
        --------
        list(str)
        """
        if sensor_bands_info is None:
            sensor_bands_info = local_context.get('sensor_bands_info')
        if sensor_bands_info is None:
            raise ProductError("sensor_bands_info must be supplied")

        return ProductsFactory.get_matchings(self.band_names, sensor_bands_info)
