import os
import json
import rasterio
import numpy as np
from rasterio import shutil as rasterio_sh
from rasterio.enums import Resampling, MaskFlags
from tempfile import TemporaryDirectory


def _calc_overviews_factors(one, blocksize=256):
    res = max(one.width, one.height)
    factor = 2
    factors = []
    while res > blocksize:
        factors.append(factor)
        res /= 2
        factor *= 2
    return factors


def _join_masks_from_masked_array(data):
    """Union of masks."""
    if not isinstance(data.mask, np.ndarray):
        # workaround to handle mask compressed to single value
        mask = np.empty(data.data.shape, dtype=np.bool)
        mask.fill(data.mask)
        return mask
    mask = data.mask[0].copy()
    for i in range(1, len(data.mask)):
        mask = np.logical_or(mask, data.mask[i])
    return mask[np.newaxis, :, :]


def _mask_from_masked_array(data):
    """Union of mask and converting from boolean to uint8.

    Numpy mask is the invers of the GDAL, True is 0 and False is 255
    https://github.com/mapbox/rasterio/blob/master/docs/topics/masks.rst#numpy-masked-arrays
    """
    mask = _join_masks_from_masked_array(data)[0]
    mask = (~mask * 255).astype('uint8')
    return mask


def _has_internal_perdataset_mask(rast):
    for flags in rast.mask_flag_enums:
        if MaskFlags.per_dataset in flags:
            return True
    return False


def _get_telluric_tags(source_file):
    with rasterio.open(source_file) as r:
        tags = r.tags(ns='rastile')
        if not tags:
            return tags
        return {"telluric_%s" % k: v for k, v in tags.items()}


def convert_to_cog(source_file, destination_file, resampling=Resampling.gauss):
    """Convert source file to a Cloud Optimized GeoTiff new file.

    :param source_file: path to the original raster
    :param destination_file: path to the new raster
    :param resampling: which Resampling to use on reading, default Resampling.gauss
    """
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp.tif')
            rasterio_sh.copy(source_file, temp_file, tiled=True, compress='DEFLATE', photometric='MINISBLACK')
            with rasterio.open(temp_file, 'r+') as dest:
                if not _has_internal_perdataset_mask(dest):
                    mask = dest.dataset_mask()
                    dest.write_mask(mask)

                factors = _calc_overviews_factors(dest)
                dest.build_overviews(factors, resampling=resampling)
                dest.update_tags(ns='rio_overview', resampling=resampling.name)

                telluric_tags = _get_telluric_tags(source_file)
                if telluric_tags:
                    dest.update_tags(**telluric_tags)

            rasterio_sh.copy(temp_file, destination_file,
                             COPY_SRC_OVERVIEWS=True, tiled=True,
                             compress='DEFLATE', photometric='MINISBLACK')
