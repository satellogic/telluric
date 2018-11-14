from tempfile import NamedTemporaryFile

from telluric import GeoRaster2
from telluric.util.raster_utils import build_vrt


def test_build_vrt():
    source_file = 'tests/data/raster/rgb.tif'
    with NamedTemporaryFile(suffix='.vrt') as fp:
        vrt = build_vrt(source_file, fp.name)
        assert GeoRaster2.open(source_file) == GeoRaster2.open(vrt)
