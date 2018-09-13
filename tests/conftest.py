import numpy as np
from affine import Affine
import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS
import pytest
from rasterio.crs import CRS


@pytest.fixture(scope="module")
def test_raster_with_no_url():
    raster = tl.GeoRaster2(np.random.uniform(0, 256, (3, 391, 370)),
                           affine=Affine(1.0000252884112817, 0.0, 2653750.345511198,
                                         0.0, -1.0000599330133702, 4594461.485763356),
                           crs=CRS({'init': 'epsg:3857'}))
    return raster


@pytest.fixture(scope="module")
def test_raster_with_url(test_raster_with_no_url):
    raster = test_raster_with_no_url
    url = "/vsimem/features.tif"
    raster.save(url)
    return tl.GeoRaster2.open(url)
