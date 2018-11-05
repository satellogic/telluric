from telluric.georaster import GeoRaster2, MutableGeoRaster
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.collections import FileCollection, FeatureCollection
import rasterio
from rasterio.crs import CRS


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

with rasterio.Env():
    CRS({'init': 'epsg:4326'}).is_geographic
