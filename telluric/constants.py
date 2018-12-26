"""Useful constants.

"""
from rasterio.crs import CRS

WGS84_SRID = 4326
#: WGS84 CRS.
WGS84_CRS = CRS.from_epsg(WGS84_SRID)

WEB_MERCATOR_SRID = 3857
#: Web Mercator CRS.
WEB_MERCATOR_CRS = CRS.from_epsg(WEB_MERCATOR_SRID)

# Best widely used, equal area projection according to
# http://icaci.org/documents/ICC_proceedings/ICC2001/icc2001/file/f24014.doc
# (found on https://en.wikipedia.org/wiki/Winkel_tripel_projection#Comparison_with_other_projections)
#: Eckert IV CRS.
EQUAL_AREA_CRS = CRS({'proj': 'eck4'})

DEFAULT_SRID = WGS84_SRID
#: Default CRS, set to :py:data:`~telluric.constants.WGS84_CRS`.
DEFAULT_CRS = WGS84_CRS


def _MERCATOR_RESOLUTION_MAPPING(zoom_level):
    return (2 * 20037508.342789244) / (256 * pow(2, zoom_level))


MERCATOR_RESOLUTION_MAPPING = dict((i, _MERCATOR_RESOLUTION_MAPPING(i)) for i in range(21))

RASTER_TYPE = 'raster'
