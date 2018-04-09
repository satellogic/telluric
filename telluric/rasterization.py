import numpy as np

from affine import Affine, TransformNotInvertibleError
from rasterio import features
from rasterio._err import CPLE_OutOfMemoryError  # noqa

from telluric.georaster import GeoRaster2


DTYPE = np.uint8
NODATA_VALUE = 0
# FILL_VALUE = np.iinfo(DTYPE).max
FILL_VALUE = 1


class ScaleError(ValueError):
    pass


def rasterize(shapes, crs, bounds, dest_resolution, fill_value=None, nodata_value=None,
              band_names=None, dtype=np.uint8):
    if fill_value is None:
        fill_value = FILL_VALUE
    if nodata_value is None:
        nodata_value = NODATA_VALUE

    if band_names is None:
        band_names = [1]

    # Affine transformation
    minx, miny, maxx, maxy = bounds.bounds
    affine = Affine.translation(minx, maxy) * Affine.scale(dest_resolution, -dest_resolution)

    # Compute size from scale
    dx = maxx - minx
    dy = maxy - miny
    sx = round(dx / dest_resolution)
    sy = round(dy / dest_resolution)
    sz = len(band_names)

    if sx == 0 or sy == 0:
        raise ScaleError("Scale is too coarse, decrease it for a bigger image")

    try:
        if not shapes:
            image = np.full((sz, sy, sx), nodata_value, dtype=dtype)

        else:
            image = features.rasterize(
                shapes,
                out_shape=(sy, sx), fill=nodata_value, transform=affine, default_value=fill_value)

    except TransformNotInvertibleError:
        raise ScaleError("Scale is too coarse, decrease it for a bigger image")

    except (MemoryError, ValueError, CPLE_OutOfMemoryError):
        raise ScaleError("Scale is too fine, increase it for a smaller image")

    else:
        return GeoRaster2(image, affine, crs, nodata=nodata_value, band_names=band_names)
