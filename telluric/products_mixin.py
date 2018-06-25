from telluric.products import ProductsFactory
from telluric.product_view import ProductViewsFactory


class ProductsMixin:
    def apply(self, sensor_bands_info, product_name, metadata=False, **kwargs):
        """Apply a product on a raster, from a pre defined list of products, such as NVDI.

        Parameters
        ----------
        sensor_bands_info: dict
            Bands wave lenght definition including max and min wave per bands name
        product_name: str
            Name of requested product
        metadata: bool
            Optional, when True it returns a namedtuple withe resulting raster and additional product information
        kwargs: dict
            Additional arguments, required to generate the poduct like the band name for SingleBand product
        
        Rerturns
        --------
        GeoRaster2
        Product()
        """
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

    def get_products(self, sensor_bands_info):
        """Return a list of product names applicable to the raster.

        Parameters
        ----------
        sensor_bands_info: dict
            Bands wave lenght definition including max and min wave per bands name
        
        Rerturns
        --------
        list(str)
        """
        return ProductsFactory.get_matchings(self.band_names, sensor_bands_info)
