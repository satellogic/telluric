from telluric.products import ProductsFactory
from telluric.product_view import ProductViewsFactory


class ProductsMixin:
    def apply(self, sensor_bands_info, product_name, metadata=False, **kwargs):
        generator = ProductsFactory.get_object(product_name, **kwargs)
        product = generator.apply(sensor_bands_info, self, metadata=metadata)
        return product

    def visualize(self, product_view_name, **kwargs):
        generator = ProductViewsFactory.get_object(product_view_name)
        view = generator.apply(self, **kwargs)
        return view

    def get_products(self, sensor_bands_info):
        return ProductsFactory.get_matchings(self.band_names, sensor_bands_info)
