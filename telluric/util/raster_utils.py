import os
import rasterio
import numpy as np
from affine import Affine
from rasterio import shutil as rasterio_sh
from rasterio.enums import Resampling, MaskFlags
from rasterio.warp import (
    transform_bounds, reproject, aligned_target,
    calculate_default_transform as calcdt)
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from math import ceil


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
        rastile_tags = r.tags(ns='rastile')
        return_tags = {}
        if rastile_tags:
            return_tags.update({"telluric_%s" % k: v for k, v in rastile_tags.items()})
        tags = r.tags()
        if tags:
            return_tags.update({k: v for k, v in tags.items() if k.startswith("telluric_")})
        return return_tags


def _creation_options_for_cog(creation_options, source_profile, blocksize):
    """
    it uses the profile of the source raster, override anything using the creation_options
    and guarantees we will have tiled raster and blocksize
    """
    if not(creation_options):
        creation_options = {}

    creation_options["blocksize"] = blocksize
    creation_options["tiled"] = True
    for key in ["nodata", "compress"]:
        if key not in creation_options:
            creation_options[key] = source_profile.get(key, None)
    return creation_options


def convert_to_cog(source_file, destination_file, resampling=rasterio.enums.Resampling.gauss, blocksize=256,
                   overview_blocksize=256, creation_options=None):
    """Convert source file to a Cloud Optimized GeoTiff new file.

    :param source_file: path to the original raster
    :param destination_file: path to the new raster
    :param resampling: which Resampling to use on reading, default Resampling.gauss
    :param blocksize: the size of the blocks default 256
    :param overview_blocksize: the block size of the overviews, default 256
    :param creation_options: <dictioanry>, options that can override the source raster profile,
                          notice that you can't override tiled=True, and the blocksize
    """

    with rasterio.open(source_file) as src:
        # creation_options overrides proile
        source_profile = src.profile
    creation_options = _creation_options_for_cog(creation_options, source_profile, blocksize)

    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE=overview_blocksize):
        with TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp.tif')
            rasterio_sh.copy(source_file, temp_file, **creation_options)
            with rasterio.open(temp_file, 'r+') as dest:
                factors = _calc_overviews_factors(dest)
                dest.build_overviews(factors, resampling=resampling)
                dest.update_tags(ns='rio_overview', resampling=resampling.name)

                telluric_tags = _get_telluric_tags(source_file)
                if telluric_tags:
                    dest.update_tags(**telluric_tags)

            rasterio_sh.copy(temp_file, destination_file,
                             COPY_SRC_OVERVIEWS=True, **creation_options)


def calc_transform(src, dst_crs=None, resolution=None, dimensions=None,
                   src_bounds=None, dst_bounds=None, target_aligned_pixels=False):
    """Output dimensions and transform for a reprojection.

    Parameters
    ------------
    src: rasterio.io.DatasetReader
        Data source.
    dst_crs: rasterio.crs.CRS, optional
        Target coordinate reference system.
    resolution: tuple (x resolution, y resolution) or float, optional
        Target resolution, in units of target coordinate reference
        system.
    dimensions: tuple (width, height), optional
        Output file size in pixels and lines.
    src_bounds: tuple (xmin, ymin, xmax, ymax), optional
        Georeferenced extent of output file from source bounds
        (in source georeferenced units).
    dst_bounds: tuple (xmin, ymin, xmax, ymax), optional
        Georeferenced extent of output file from destination bounds
        (in destination georeferenced units).
    target_aligned_pixels: bool, optional
        Align the output bounds based on the resolution.
        Default is `False`.

    Returns
    -------
    dst_crs: rasterio.crs.CRS
        Output crs
    transform: Affine
        Output affine transformation matrix
    width, height: int
        Output dimensions
    """
    if resolution is not None:
        if isinstance(resolution, (float, int)):
            resolution = (float(resolution), float(resolution))

    if target_aligned_pixels:
        if not resolution:
            raise ValueError('target_aligned_pixels cannot be used without resolution')
        if src_bounds or dst_bounds:
            raise ValueError('target_aligned_pixels cannot be used with src_bounds or dst_bounds')

    elif dimensions:
        invalid_combos = (dst_bounds, resolution)
        if any(p for p in invalid_combos if p is not None):
            raise ValueError('dimensions cannot be used with dst_bounds or resolution')

    if src_bounds and dst_bounds:
        raise ValueError('src_bounds and dst_bounds may not be specified simultaneously')

    if dst_crs is not None:

        if dimensions:
            # Calculate resolution appropriate for dimensions
            # in target.
            dst_width, dst_height = dimensions
            bounds = src_bounds or src.bounds
            xmin, ymin, xmax, ymax = transform_bounds(
                src.crs, dst_crs, *bounds)
            dst_transform = Affine(
                (xmax - xmin) / float(dst_width),
                0, xmin, 0,
                (ymin - ymax) / float(dst_height),
                ymax
            )

        elif src_bounds or dst_bounds:
            if not resolution:
                raise ValueError('resolution is required when using src_bounds or dst_bounds')

            if src_bounds:
                xmin, ymin, xmax, ymax = transform_bounds(
                    src.crs, dst_crs, *src_bounds)
            else:
                xmin, ymin, xmax, ymax = dst_bounds

            dst_transform = Affine(resolution[0], 0, xmin, 0, -resolution[1], ymax)
            dst_width = max(int(ceil((xmax - xmin) / resolution[0])), 1)
            dst_height = max(int(ceil((ymax - ymin) / resolution[1])), 1)

        else:
            if src.transform.is_identity and src.gcps:
                src_crs = src.gcps[1]
                kwargs = {'gcps': src.gcps[0]}
            else:
                src_crs = src.crs
                kwargs = src.bounds._asdict()
            dst_transform, dst_width, dst_height = calcdt(
                src_crs, dst_crs, src.width, src.height,
                resolution=resolution, **kwargs)

    elif dimensions:
        # Same projection, different dimensions, calculate resolution.
        dst_crs = src.crs
        dst_width, dst_height = dimensions
        l, b, r, t = src_bounds or src.bounds
        dst_transform = Affine(
            (r - l) / float(dst_width),
            0, l, 0,
            (b - t) / float(dst_height),
            t
        )

    elif src_bounds or dst_bounds:
        # Same projection, different dimensions and possibly
        # different resolution.
        if not resolution:
            resolution = (src.transform.a, -src.transform.e)

        dst_crs = src.crs
        xmin, ymin, xmax, ymax = (src_bounds or dst_bounds)
        dst_transform = Affine(resolution[0], 0, xmin, 0, -resolution[1], ymax)
        dst_width = max(int(ceil((xmax - xmin) / resolution[0])), 1)
        dst_height = max(int(ceil((ymax - ymin) / resolution[1])), 1)

    elif resolution:
        # Same projection, different resolution.
        dst_crs = src.crs
        l, b, r, t = src.bounds
        dst_transform = Affine(resolution[0], 0, l, 0, -resolution[1], t)
        dst_width = max(int(ceil((r - l) / resolution[0])), 1)
        dst_height = max(int(ceil((t - b) / resolution[1])), 1)

    else:
        dst_crs = src.crs
        dst_transform = src.transform
        dst_width = src.width
        dst_height = src.height

    if target_aligned_pixels:
        dst_transform, dst_width, dst_height = aligned_target(
            dst_transform, dst_width, dst_height, resolution)

    return dst_crs, dst_transform, dst_width, dst_height


# Code was adapted from rasterio.rio.warp module
def warp(source_file, destination_file, dst_crs=None, resolution=None, dimensions=None,
         src_bounds=None, dst_bounds=None, src_nodata=None, dst_nodata=None,
         target_aligned_pixels=False, check_invert_proj=True,
         creation_options=None, resampling=Resampling.cubic, **kwargs):
    """Warp a raster dataset.

    Parameters
    ------------
    source_file: str, file object or pathlib.Path object
        Source file.
    destination_file: str, file object or pathlib.Path object
        Destination file.
    dst_crs: rasterio.crs.CRS, optional
        Target coordinate reference system.
    resolution: tuple (x resolution, y resolution) or float, optional
        Target resolution, in units of target coordinate reference
        system.
    dimensions: tuple (width, height), optional
        Output file size in pixels and lines.
    src_bounds: tuple (xmin, ymin, xmax, ymax), optional
        Georeferenced extent of output file from source bounds
        (in source georeferenced units).
    dst_bounds: tuple (xmin, ymin, xmax, ymax), optional
        Georeferenced extent of output file from destination bounds
        (in destination georeferenced units).
    src_nodata: int, float, or nan, optional
        Manually overridden source nodata.
    dst_nodata: int, float, or nan, optional
        Manually overridden destination nodata.
    target_aligned_pixels: bool, optional
        Align the output bounds based on the resolution.
        Default is `False`.
    check_invert_proj: bool, optional
        Constrain output to valid coordinate region in dst_crs.
        Default is `True`.
    creation_options: dict, optional
        Custom creation options.
    resampling: rasterio.enums.Resampling
        Reprojection resampling method. Default is `cubic`.
    kwargs: optional
        Additional arguments passed to transformation function.

    Returns
    ---------
    out: None
        Output is written to destination.
    """

    with rasterio.Env(CHECK_WITH_INVERT_PROJ=check_invert_proj):
        with rasterio.open(source_file) as src:
            out_kwargs = src.profile.copy()
            dst_crs, dst_transform, dst_width, dst_height = calc_transform(
                src, dst_crs, resolution, dimensions,
                src_bounds, dst_bounds, target_aligned_pixels)

            # If src_nodata is not None, update the dst metadata NODATA
            # value to src_nodata (will be overridden by dst_nodata if it is not None.
            if src_nodata is not None:
                # Update the destination NODATA value
                out_kwargs.update({
                    'nodata': src_nodata
                })

            # Validate a manually set destination NODATA value.
            if dst_nodata is not None:
                if src_nodata is None and src.meta['nodata'] is None:
                    raise ValueError('src_nodata must be provided because dst_nodata is not None')
                else:
                    out_kwargs.update({'nodata': dst_nodata})

            out_kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })

            # Adjust block size if necessary.
            if ('blockxsize' in out_kwargs and
                    dst_width < out_kwargs['blockxsize']):
                del out_kwargs['blockxsize']
            if ('blockysize' in out_kwargs and
                    dst_height < out_kwargs['blockysize']):
                del out_kwargs['blockysize']

            if creation_options is not None:
                out_kwargs.update(**creation_options)

            with rasterio.open(destination_file, 'w', **out_kwargs) as dst:
                reproject(
                    source=rasterio.band(src, src.indexes),
                    destination=rasterio.band(dst, dst.indexes),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=out_kwargs['transform'],
                    dst_crs=out_kwargs['crs'],
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                    **kwargs)


def build_overviews(source_file, factors=None, minsize=256, external=False,
                    blocksize=256, interleave='pixel', compress='lzw',
                    resampling=Resampling.gauss, **kwargs):
    """Build overviews at one or more decimation factors for all
    bands of the dataset.

    Parameters
    ------------
    source_file : str, file object or pathlib.Path object
        Source file.
    factors : list, optional
        A list of integral overview levels to build.
    minsize : int, optional
        Maximum width or height of the smallest overview level. Only taken into account
        if explicit factors are not specified. Defaults to `256`.
    external : bool, optional
        Can be set to `True` to force external overviews in the GeoTIFF (.ovr) format.
        Default is False.
    blocksize : int, optional
        The block size (tile width and height) used for overviews. Should be a
        power-of-two value between 64 and 4096. Default value is `256`.
    interleave : str, optional
        Interleaving. Default value is `pixel`.
    compress : str, optional
        Set the compression to use. Default is `lzw`.
    resampling : rasterio.enums.Resampling
        Resampling method. Default is `gauss`.
    kwargs : optional
        Additional arguments passed to rasterio.Env.

    Returns
    ---------
    out: None
        Original file is altered or external .ovr can be created.
    """

    with rasterio.open(source_file, 'r+') as dst:
        if factors is None:
            factors = _calc_overviews_factors(
                SimpleNamespace(width=dst.width, height=dst.height), minsize)

        with rasterio.Env(
            GDAL_TIFF_OVR_BLOCKSIZE=blocksize,
            INTERLEAVE_OVERVIEW=interleave,
            COMPRESS_OVERVIEW=compress,
            TIFF_USE_OVR=external,
            **kwargs
        ):
            dst.build_overviews(factors, resampling)
