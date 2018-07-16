import os
import rasterio
import numpy as np
from rasterio import shutil as rasterio_sh
from rasterio.enums import Resampling, MaskFlags
from rasterio.warp import calculate_default_transform
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


def convert_to_cog(source_file, destination_file, resampling=Resampling.gauss, **kwargs):
    """Convert source file to a Cloud Optimized GeoTiff new file.

    :param source_file: path to the original raster
    :param destination_file: path to the new raster
    :param resampling: which Resampling to use on reading, default Resampling.gauss
    """
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE=256):
    #with rasterio.Env(GDAL_TIFF_OVR_BLOCKSIZE=256):
        with TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp.tif')
            rasterio_sh.copy(source_file, temp_file, tiled=True, compress='DEFLATE', photometric='MINISBLACK')
            with rasterio.open(temp_file, 'r+') as dest:
                # if not _has_internal_perdataset_mask(dest):
                    # mask = dest.dataset_mask()
                    # dest.write_mask(mask)

                factors = _calc_overviews_factors(dest)
                dest.build_overviews(factors, resampling=resampling)
                dest.update_tags(ns='rio_overview', resampling=resampling.name)

                telluric_tags = _get_telluric_tags(source_file)
                if telluric_tags:
                    dest.update_tags(**telluric_tags)
            rasterio_sh.copy(temp_file, destination_file,
                             COPY_SRC_OVERVIEWS=True, tiled=True,
                             compress='DEFLATE', photometric='MINISBLACK')


def reproject(source_file, destination_file, crs, resolution=None, profile=None, **kwargs):
    """Reproject a source file to a destination file.

    :param source_file: path to the original raster
    :param destination_file: path to the new raster
    :param crs: target coordinate reference system
    :param resolution: target resolution, in units of target crs
    :param profile: custom creation options
    :param kwargs: additional arguments passed to transformation function
    """
    with rasterio.Env():
        with rasterio.open(source_file) as src:
            affine, width, height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds,
                resolution=resolution)

            sprofile = src.profile.copy()
            sprofile.update({
                'crs': crs,
                'transform': affine,
                'width': width,
                'height': height})
            if profile is not None:
                sprofile.update(profile)

            with rasterio.open(destination_file, 'w', **sprofile) as dst:
                rasterio.warp.reproject(
                    rasterio.band(src, src.indexes),
                    rasterio.band(dst, dst.indexes),
                    **kwargs)
