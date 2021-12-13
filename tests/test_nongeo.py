import os
from pathlib import Path
import numpy as np
import pytest
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning
from telluric.constants import DEFAULT_CRS
from telluric import GeoRaster2


def test_save_nongeo(tmp_path, recwarn):
    raster = GeoRaster2(image=np.ones([10, 10], dtype=np.uint8),
                        crs=DEFAULT_CRS, affine=Affine.translation(10, 100))
    path = tmp_path / 'raster1.tif'
    raster.save(path)
    raster = GeoRaster2.open(path)
    assert raster.crs is not None and raster.affine is not None

    raster = raster.copy_with(crs=None, affine=None)
    assert raster.crs is None and raster.affine is None
    path = Path('/tmp/raster2.tif')
    raster.save(path)

    raster = GeoRaster2.open(path)
    assert raster.crs is None
    assert raster.affine == Affine.identity()  # rasterio does this
