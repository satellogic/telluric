import json
import os
import io
import contextlib
from functools import reduce, partial
from typing import Callable, Union, Iterable, Dict, List, Optional, Tuple
from types import SimpleNamespace
from enum import Enum

import tempfile
from copy import copy, deepcopy

import math
from itertools import groupby

import mercantile

import warnings

import numpy as np
import imageio

from boltons.setutils import IndexedSet

from rasterio.crs import CRS
import rasterio
import rasterio.warp
import rasterio.shutil
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling, Compression
from rasterio.features import geometry_mask
from rasterio.windows import Window, WindowMethodsMixin
from affine import Affine

from shapely.geometry import Point, Polygon

from PIL import Image

from telluric.constants import DEFAULT_CRS, WEB_MERCATOR_CRS, MERCATOR_RESOLUTION_MAPPING
from telluric.vectors import GeoVector
from telluric.util.projections import transform

from telluric.util.raster_utils import (
    convert_to_cog, _calc_overviews_factors,
    _mask_from_masked_array, _join_masks_from_masked_array,
    calc_transform, warp)

import matplotlib  # for mypy

with warnings.catch_warnings():  # silences warning, see https://github.com/matplotlib/matplotlib/issues/5836
    warnings.simplefilter("ignore", UserWarning)
    import matplotlib.pyplot as plt


dtype_map = {
    np.uint8: rasterio.uint8,
    np.uint16: rasterio.uint16,
    np.uint32: rasterio.uint32,
    np.int16: rasterio.int16,
    np.int32: rasterio.int32,
    np.float32: rasterio.float32,
    np.float64: rasterio.float64,
}


gdal_drivers = {
    'tif': 'GTiff',
    'tiff': 'GTiff',
    'png': 'PNG',
    'jpg': 'JPEG',
    'jpeg': 'JPEG',
}


class MergeStrategy(Enum):
    LEFT_ALL = 0
    INTERSECTION = 1
    UNION = 2


class PixelStrategy(Enum):
    INDEX = 0
    FIRST = 1


def merge_all(rasters, roi=None, dest_resolution=None, merge_strategy=MergeStrategy.UNION,
              shape=None, ul_corner=None, crs=None,
              pixel_strategy=PixelStrategy.FIRST):
    """Merge a list of rasters, cropping by a region of interest.
       There are cases that the roi is not precise enough for this cases one can use,
       the upper left corner the shape and crs to precisely define the roi.
       When roi is provided the ul_corner, shape and crs are ignored
    """
    if dest_resolution is None:
        dest_resolution = rasters[0].res_xy()

    # Create empty raster
    empty = GeoRaster2.empty_from_roi(
        roi, resolution=dest_resolution, band_names=rasters[0].band_names,
        dtype=rasters[0].dtype, shape=shape, ul_corner=ul_corner, crs=crs)

    # Create a list of single band rasters
    all_band_names, projected_rasters = _prepare_rasters(rasters, merge_strategy, empty)
    assert len(projected_rasters) == len(rasters)

    prepared_rasters = _apply_pixel_strategy(projected_rasters, pixel_strategy)

    # Extend the rasters list with only those that have the requested bands
    prepared_rasters = _explode_rasters(prepared_rasters, all_band_names)

    if all_band_names:
        # Merge common bands
        prepared_rasters = _merge_common_bands(prepared_rasters)

        # Merge all bands
        raster = reduce(_stack_bands, prepared_rasters)

        return empty.copy_with(image=raster.image, band_names=raster.band_names)

    else:
        raise ValueError("result contains no bands, use another merge strategy")


def _apply_pixel_strategy(rasters, pixel_strategy):
    # type: (List[Optional[_Raster]], PixelStrategy) -> List[_Raster]
    if pixel_strategy == PixelStrategy.INDEX:
        new_rasters = []
        for ii, raster in enumerate(rasters):
            if raster:
                new_image = np.ma.masked_array(
                    np.full_like(raster.image.data, ii, dtype=int),
                    raster.image.mask
                )
                new_rasters.append(_Raster(image=new_image, band_names=raster.band_names))

        return new_rasters

    elif pixel_strategy == PixelStrategy.FIRST:
        # The way merge_all is written now, this pixel strategy is the default one
        # and all the steps in the chain are prepared for it, so no changes needed
        # apart from taking out None values
        return [raster for raster in rasters if raster]

    else:
        raise ValueError("Please use an allowed pixel_strategy")


def _explode_rasters(projected_rasters, all_band_names):
    # type: (List[_Raster], IndexedSet[str]) -> List[_Raster]
    prepared_rasters = []
    for projected_raster in projected_rasters:
        prepared_rasters.extend(_explode_raster(projected_raster, all_band_names))

    return prepared_rasters


def _merge_common_bands(rasters):
    # type: (List[_Raster]) -> List[_Raster]
    """Combine the common bands.

    """
    # Compute band order
    all_bands = IndexedSet([rs.band_names[0] for rs in rasters])

    def key(rs):
        return all_bands.index(rs.band_names[0])

    rasters_final = []  # type: List[_Raster]
    for band_name, rasters_group in groupby(sorted(rasters, key=key), key=key):
        rasters_final.append(reduce(_fill_pixels, rasters_group))

    return rasters_final


def _prepare_rasters(rasters, merge_strategy, first):
    # type: (List[GeoRaster2], MergeStrategy, GeoRaster2) -> Tuple[IndexedSet[str], List[Optional[_Raster]]]
    """Prepares the rasters according to the baseline (first) raster and the merge strategy.

    The baseline (first) raster is used to crop and reproject the other rasters,
    while the merge strategy is used to compute the bands of the result. These
    are returned for diagnostics.

    """
    # Create list of prepared rasters
    all_band_names = IndexedSet(first.band_names)
    projected_rasters = []
    for raster in rasters:
        projected_raster = _prepare_other_raster(first, raster)

        # Modify the bands only if an intersecting raster was returned
        if projected_raster:
            if merge_strategy is MergeStrategy.INTERSECTION:
                all_band_names.intersection_update(projected_raster.band_names)
            elif merge_strategy is MergeStrategy.UNION:
                all_band_names.update(projected_raster.band_names)

        # Some rasters might be None. In this way, we still retain the original order
        projected_rasters.append(projected_raster)

    return all_band_names, projected_rasters


# noinspection PyDefaultArgument
def _explode_raster(raster, band_names=[]):
    # type: (_Raster, Iterable[str]) -> List[_Raster]
    """Splits a raster into multiband rasters.

    """
    # Using band_names=[] does no harm because we are not mutating it in place
    # and it makes MyPy happy
    if not band_names:
        band_names = raster.band_names
    else:
        band_names = list(IndexedSet(raster.band_names).intersection(band_names))

    return [_Raster(image=raster.bands_data([band_name]), band_names=[band_name]) for band_name in band_names]


def _prepare_other_raster(one, other):
    # type: (GeoRaster2, GeoRaster2) -> Union[_Raster, None]
    # Crop and reproject the second raster, if necessary
    if not (one.crs == other.crs and one.affine.almost_equals(other.affine) and one.shape == other.shape):
        if one.footprint().intersects(other.footprint()):
            if one.crs != other.crs:
                src_bounds = one.footprint().get_bounds(other.crs)
                src_vector = GeoVector(Polygon.from_bounds(*src_bounds), other.crs)
                src_width, src_height = (
                    src_bounds.right - src_bounds.left,
                    src_bounds.top - src_bounds.bottom)
                buffer_ratio = int(os.environ.get("TELLURIC_MERGE_CROP_BUFFER", 10))
                buffer_size = max(src_width, src_height) * (buffer_ratio / 100)
                other = other.crop(src_vector.buffer(buffer_size))
            else:
                other = other.crop(one.footprint(), resolution=one.resolution())

            other = other._reproject(new_width=one.width, new_height=one.height,
                                     dest_affine=one.affine, dst_crs=one.crs,
                                     resampling=Resampling.nearest)

        else:
            return None

    return _Raster(image=other.image, band_names=other.band_names)


def _fill_pixels(one, other):
    # type: (_Raster, _Raster) -> _Raster
    """Merges two single band rasters with the same band by filling the pixels according to depth.

    """
    assert len(one.band_names) == len(other.band_names) == 1, "Rasters are not single band"

    # We raise an error in the intersection is empty.
    # Other options include returning an "empty" raster or just None.
    # The problem with the former is that GeoRaster2 expects a 2D or 3D
    # numpy array, so there is no obvious way to signal that this raster
    # has no bands. Also, returning a (1, 1, 0) numpy array is useless
    # for future concatenation, so the expected shape should be used
    # instead. The problem with the latter is that it breaks concatenation
    # anyway and requires special attention. Suggestions welcome.
    if one.band_names != other.band_names:
        raise ValueError("rasters have no bands in common, use another merge strategy")

    new_image = one.image.copy()
    other_image = other.image

    # The values that I want to mask are the ones that:
    # * Were already masked in the other array, _or_
    # * Were already unmasked in the one array, so I don't overwrite them
    other_values_mask = (np.ma.getmaskarray(other_image)[0] | (~np.ma.getmaskarray(one.image)[0]))

    # Reshape the mask to fit the future array
    other_values_mask = other_values_mask[None, ...]

    # Overwrite the values that I don't want to mask
    new_image[~other_values_mask] = other_image[~other_values_mask]

    # In other words, the values that I wanted to write are the ones that:
    # * Were already masked in the one array, _and_
    # * Were not masked in the other array
    # The reason for using the inverted form is to retain the semantics
    # of "masked=True" that apply for masked arrays. The same logic
    # could be written, using the De Morgan's laws, as
    # other_values_mask = (one.image.mask[0] & (~other_image.mask[0])
    # other_values_mask = other_values_mask[None, ...]
    # new_image[other_values_mask] = other_image[other_values_mask]
    # but here the word "mask" does not mean the same as in masked arrays.

    return _Raster(image=new_image, band_names=one.band_names)


def _stack_bands(one, other):
    # type: (_Raster, _Raster) -> _Raster
    """Merges two rasters with non overlapping bands by stacking the bands.

    """
    assert set(one.band_names).intersection(set(other.band_names)) == set()

    # We raise an error in the bands are the same. See above.
    if one.band_names == other.band_names:
        raise ValueError("rasters have the same bands, use another merge strategy")

    # Apply "or" to the mask in the same way rasterio does, see
    # https://mapbox.github.io/rasterio/topics/masks.html#dataset-masks
    # In other words, mask the values that are already masked in either
    # of the two rasters, since one mask per band is not supported
    new_mask = np.ma.getmaskarray(one.image)[0] | np.ma.getmaskarray(other.image)[0]

    # Concatenate the data along the band axis and apply the mask
    new_image = np.ma.masked_array(
        np.concatenate([
            one.image.data,
            other.image.data
        ]),
        mask=[new_mask] * (one.image.shape[0] + other.image.shape[0])
    )
    new_bands = one.band_names + other.band_names

    # We don't copy image and mask here, due to performance issues,
    # this output should not use without eventually being copied
    # In this context we are copying the object in the end of merge_all merge_first and merge
    return _Raster(image=new_image, band_names=new_bands)


def merge_two(one, other, merge_strategy=MergeStrategy.UNION, silent=False, pixel_strategy=PixelStrategy.FIRST):
    # type: (GeoRaster2, GeoRaster2, MergeStrategy, bool, PixelStrategy) -> GeoRaster2
    """Merge two rasters into one.

    Parameters
    ----------
    one : GeoRaster2
        Left raster to merge.
    other : GeoRaster2
        Right raster to merge.
    merge_strategy : MergeStrategy, optional
        Merge strategy, from :py:data:`telluric.georaster.MergeStrategy` (default to "union").
    silent : bool, optional
        Whether to raise errors or return some result, default to False (raise errors).
    pixel_strategy: PixelStrategy, optional
        Pixel strategy, from :py:data:`telluric.georaster.PixelStrategy` (default to "top").

    Returns
    -------
    GeoRaster2

    """
    other_res = _prepare_other_raster(one, other)
    if other_res is None:
        if silent:
            return one
        else:
            raise ValueError("rasters do not intersect")

    else:
        other = other.copy_with(image=other_res.image, band_names=other_res.band_names)  # To make MyPy happy

    # Create a list of single band rasters
    # Cropping won't happen twice, since other was already cropped
    all_band_names, projected_rasters = _prepare_rasters([other], merge_strategy, first=one)

    if not all_band_names and not silent:
        raise ValueError("rasters have no bands in common, use another merge strategy")

    prepared_rasters = _apply_pixel_strategy(projected_rasters, pixel_strategy)

    prepared_rasters = _explode_rasters(prepared_rasters, all_band_names)

    # Merge common bands
    prepared_rasters = _merge_common_bands(_explode_raster(one, all_band_names) + prepared_rasters)

    # Merge all bands
    raster = reduce(_stack_bands, prepared_rasters)

    return one.copy_with(image=raster.image, band_names=raster.band_names)


class GeoRaster2Warning(UserWarning):
    """Base class for warnings in the GeoRaster class."""
    pass


class GeoRaster2Error(Exception):
    """Base class for exceptions in the GeoRaster class."""

    pass


class GeoRaster2IOError(GeoRaster2Error):
    """Base class for exceptions in GeoRaster read/write."""

    pass


class GeoRaster2NotImplementedError(GeoRaster2Error, NotImplementedError):
    """Base class for NotImplementedError in the GeoRaster class. """
    pass


class _Raster:
    """ A class that has image, band_names and shape
    """

    def __init__(self, image=None, band_names=None, shape=None, nodata=0):
        """Create a GeoRaster object

        :param image: optional supported: np.ma.array, np.array, TODO: PIL image
        :param band_names: e.g. ['red', 'blue'] or 'red'
        :param shape: raster image shape, optional
        """
        self._image = None
        self._band_names = None
        self._shape = copy(shape)
        if band_names:
            self._set_bandnames(copy(band_names))
        if image is not None:
            self._set_image(image.copy(), nodata)
            self._dtype = np.dtype(image.dtype)
        else:
            self._dtype = None

    def _set_image(self, image, nodata=0):
        """
        Set self._image.

        :param image: supported: np.ma.array, np.array, TODO: PIL image
        :param nodata: if provided image is array (not masked array), treat pixels with value=nodata as nodata
        :return:
        """
        # convert to masked array:
        if isinstance(image, np.ma.core.MaskedArray):
            masked = image
        elif isinstance(image, np.core.ndarray):
            masked = np.ma.masked_array(image, image == nodata)
        else:
            raise GeoRaster2NotImplementedError('only ndarray or masked array supported, got %s' % type(image))

        # make sure array is 3d:
        if len(masked.shape) == 3:
            self._image = masked
        elif len(masked.shape) == 2:
            self._image = masked[np.newaxis, :, :]
        else:
            raise GeoRaster2Error('expected 2d or 3d image, got shape=%s' % masked.shape)

        # update shape
        if self._shape is None:
            self._set_shape(self._image.shape)

        self._image_after_load_validations()

        self._image.setflags(write=0)

    def _set_shape(self, shape):
        self._shape = shape
        # update band_names
        if self._band_names is None:
            self._set_bandnames(list(range(self._shape[0])))

    def _image_after_load_validations(self):
        if self._image is None:
            return
        if self._shape != self._image.shape:
            raise GeoRaster2Error('image.shape and self.shape are not equal, image.shape=%s, self.shape=%s' %
                                  (self._image.shape, self._shape))

        if self._shape[0] != len(self._band_names):
            raise GeoRaster2Error("Expected %s bands, found %s." % (len(self._band_names), self.image.shape[0]))

    def _set_bandnames(self, band_names=None):
        if isinstance(band_names, str):  # single band:
            band_names = [band_names]
        self._band_names = list(band_names)

    def bands_data(self, bands):
        if isinstance(bands, str):
            bands = bands.split(",")

        missing_bands = set(bands) - set(self.band_names)
        if missing_bands:
            raise GeoRaster2Error('requested bands %s that are not found in raster' % missing_bands)

        bands_indices = [self.band_names.index(band) for band in bands]
        bands_data = self.image[bands_indices, :, :]
        return bands_data

    @property
    def band_names(self):
        return self._band_names or []

    @property
    def image(self):
        return self._image


def resolution_from_affine(affine):
    return float(np.sqrt(np.abs(affine.determinant)))


class GeoRaster2(WindowMethodsMixin, _Raster):
    """
    Represents multiband georeferenced image, supporting nodata pixels.
    The name "GeoRaster2" is temporary.

    conventions:

    * .array is np.masked_array, mask=True on nodata pixels.
    * .array is [band, y, x]
    * .affine is affine.Affine
    * .crs is rasterio.crs.CRS
    * .band_names is list of strings, order corresponding to order in .array

    """
    def __init__(self, image=None, affine=None, crs=None,
                 filename=None, band_names=None, nodata=0, shape=None, footprint=None,
                 temporary=False):
        """Create a GeoRaster object

        :param filename: optional path/url to raster file for lazy loading
        :param image: optional supported: np.ma.array, np.array, TODO: PIL image
        :param affine: affine.Affine, or 9 numbers:
            [step_x, 0, origin_x, 0, step_y, origin_y, 0, 0, 1]
        :param crs: wkt/epsg code, e.g. {'init': 'epsg:32620'}
        :param band_names: e.g. ['red', 'blue'] or 'red'
        :param shape: raster image shape, optional
        :param nodata: if provided image is array (not masked array), treat pixels with value=nodata as nodata
        :param temporary: True means that file referenced by filename is temporary
            and will be removed by destructor, default False
        """
        super().__init__(image=image, band_names=band_names, shape=shape, nodata=nodata)
        self._affine = deepcopy(affine)
        self._crs = CRS(copy(crs)) if crs else None  # type: Union[None, CRS]
        self._filename = filename
        self._temporary = temporary
        self._footprint = copy(footprint)

    def __del__(self):
        try:
            self._cleanup()

        except Exception:
            pass

    @staticmethod
    def get_gdal_env(url):
        ret = {}
        proxy = os.environ.get("TELLURIC_HTTP_PROXY")
        if url.startswith("http://") and proxy:
            ret["GDAL_HTTP_PROXY"] = proxy
        return ret

    #  IO:
    @classmethod
    def _raster_opener(cls, filename, *args, **kwargs):
        """Return handler to open rasters (rasterio.open)."""
        with rasterio.Env(**cls.get_gdal_env(filename)):
            try:
                return rasterio.open(filename, *args, **kwargs)
            except (rasterio.errors.RasterioIOError, rasterio._err.CPLE_BaseError) as e:
                raise GeoRaster2IOError(e)

    @classmethod
    def open(cls, filename, band_names=None, lazy_load=True, **kwargs):
        """
        Read a georaster from a file.

        :param filename: url
        :param band_names: list of strings, or string.
                            if None - will try to read from image, otherwise - these will be ['0', ..]
        :param lazy_load: if True - do not load anything
        :return: GeoRaster2
        """
        geo_raster = cls(filename=filename, band_names=band_names, **kwargs)
        if not lazy_load:
            geo_raster._populate_from_rasterio_object(read_image=True)
        return geo_raster

    @classmethod
    def empty_from_roi(cls, roi=None, resolution=None,
                       band_names=None, dtype=np.uint8, ul_corner=None, shape=None, crs=None):
        from telluric import rasterization

        if roi:
            crs = crs or roi.crs
            roi = roi.get_shape(crs)

        return rasterization.rasterize([], crs, roi, resolution, band_names=band_names,
                                       dtype=dtype, shape=shape, ul_corner=ul_corner)

    def _cleanup(self):
        if self._filename is not None and self._temporary:
            with contextlib.suppress(FileNotFoundError):
                os.remove(self._filename)
            self._filename = None
            self._temporary = False

    def _populate_from_rasterio_object(self, read_image):
        with self._raster_opener(self._filename) as raster:  # type: rasterio.DatasetReader
            self._dtype = np.dtype(raster.dtypes[0])

            if self._affine is None:
                self._affine = copy(raster.transform)

            if self._crs is None:
                self._crs = copy(raster.crs)

            assert self._crs.is_valid

            # if band_names not provided, try read them from raster tags.
            # if not - leave empty, for default:
            key_name = None
            if self._band_names is None:
                tags = raster.tags(ns="rastile")
                band_names = None
                if "band_names" in tags:
                    key_name = 'band_names'

                else:
                    tags = raster.tags()
                    if tags and 'telluric_band_names' in tags:
                        key_name = 'telluric_band_names'

                if key_name is not None:
                    band_names = tags[key_name]
                    if isinstance(band_names, str):
                        band_names = json.loads(band_names)
                    self._set_bandnames(band_names)

            if read_image:
                image = np.ma.masked_array(raster.read(), ~raster.read_masks()).copy()
                self._set_image(image)
            else:
                self._set_shape((raster.count, raster.shape[0], raster.shape[1]))

    @classmethod
    def tags(cls, filename, namespace=None):
        """Extract tags from file."""
        return cls._raster_opener(filename).tags(ns=namespace)

    @property
    def image(self):
        """Raster bitmap in numpy array."""
        if self._image is None:
            self._populate_from_rasterio_object(read_image=True)
        return self._image

    @property
    def band_names(self):
        """Raster affine."""
        if self._band_names is None:
            self._populate_from_rasterio_object(read_image=False)
        return self._band_names

    @property
    def affine(self):
        """Raster affine."""
        if self._affine is None:
            self._populate_from_rasterio_object(read_image=False)
        return self._affine

    transform = affine

    @property
    def crs(self):  # type: () -> CRS
        """Raster crs."""
        if self._crs is None:
            self._populate_from_rasterio_object(read_image=False)
        return self._crs

    @property
    def shape(self):
        """Raster shape."""
        if self._shape is None:
            self._populate_from_rasterio_object(read_image=False)
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            self._populate_from_rasterio_object(read_image=False)
        return self._dtype

    @property
    def num_bands(self):
        """Raster number of bands."""
        return int(self.shape[0])

    @property
    def width(self):
        """Raster width."""
        return int(self.shape[2])

    @property
    def height(self):
        """Raster height."""
        return int(self.shape[1])

    def save(self, filename, tags=None, **kwargs):
        """
        Save GeoRaster to a file.

        :param filename: url
        :param tags: tags to add to default namespace

        optional parameters:

        * GDAL_TIFF_INTERNAL_MASK: specifies whether mask is within image file, or additional .msk
        * overviews: if True, will save with previews. default: True
        * factors: list of factors for the overview, default: calculated based on raster width and height
        * resampling: to build overviews. default: cubic
        * tiled: if True raster will be saved tiled, default: False
        * compress: any supported rasterio.enums.Compression value, default to LZW
        * blockxsize: int, tile x size, default:256
        * blockysize: int, tile y size, default:256
        * creation_options: dict, key value of additional creation options
        * nodata: if passed, will save with nodata value (e.g. useful for qgis)

        """
        if not filename.startswith("/vsi"):
            folder = os.path.abspath(os.path.join(filename, os.pardir))
            os.makedirs(folder, exist_ok=True)

        if (
            (self._image is None and self._filename is not None) and
            (tags is None and not kwargs)
        ):
            rasterio.shutil.copy(self._filename, filename)
            self._cleanup()
            self._filename = filename
            return

        internal_mask = kwargs.get('GDAL_TIFF_INTERNAL_MASK', True)
        nodata_value = kwargs.get('nodata', None)
        compression = kwargs.get('compression', Compression.lzw)
        rasterio_envs = {'GDAL_TIFF_INTERNAL_MASK': internal_mask}
        if os.environ.get('DEBUG', False):
            rasterio_envs['CPL_DEBUG'] = True
        with rasterio.Env(**rasterio_envs):
            try:
                size = self.image.shape
                extension = os.path.splitext(filename)[1].lower()[1:]
                driver = gdal_drivers[extension]

                # tiled
                tiled = kwargs.get('tiled', False)
                blockxsize = kwargs.get('blockxsize', 256)
                blockysize = kwargs.get('blockysize', 256)

                params = {
                    'mode': "w", 'transform': self.affine, 'crs': self.crs,
                    'driver': driver, 'width': size[2], 'height': size[1], 'count': size[0],
                    'dtype': dtype_map[self.image.dtype.type],
                    'nodata': nodata_value,
                    'masked': True,
                    'blockxsize': min(blockxsize, size[2]),
                    'blockysize': min(blockysize, size[1]),
                    'tiled': tiled,
                    'compress': compression.name if compression in Compression else compression,
                }
                # additional creation options
                # -co COPY_SRC_OVERVIEWS=YES  -co COMPRESS=DEFLATE -co PHOTOMETRIC=MINISBLACK
                creation_options = kwargs.get('creation_options', {})
                if creation_options:
                    params.update(**creation_options)

                with self._raster_opener(filename, **params) as r:

                    # write data:
                    for band in range(self.image.shape[0]):
                        if nodata_value is not None:
                            img = deepcopy(self.image)
                            # those pixels aren't nodata, make sure they're not set to nodata:
                            img.data[np.logical_and(img == nodata_value, self.image.mask is False)] = nodata_value + 1
                            img = np.ma.filled(img, nodata_value)
                        else:
                            img = self.image.data
                        r.write_band(1 + band, img[band, :, :])

                    # write mask:
                    if not (
                        np.ma.getmaskarray(self.image) ==
                        np.ma.getmaskarray(self.image)[0]
                    ).all():
                        warnings.warn(
                            "Saving different masks per band is not supported, "
                            "the union of the masked values will be performed.", GeoRaster2Warning
                        )

                    mask = _mask_from_masked_array(self.image)
                    r.write_mask(mask)

                    # write tags:
                    tags_to_save = {'telluric_band_names': json.dumps(self.band_names)}
                    if tags:
                        tags_to_save.update(tags)

                    r.update_tags(**tags_to_save)  # in default namespace

                    # overviews:
                    overviews = kwargs.get('overviews', True)
                    resampling = kwargs.get('resampling', Resampling.cubic)
                    if overviews:
                        factors = kwargs.get('factors')
                        if factors is None:
                            factors = self._overviews_factors()
                        else:
                            factor_max = max(self._overviews_factors(blocksize=1), default=0)
                            factors = [f for f in factors if f <= factor_max]
                        r.build_overviews(factors, resampling=resampling)
                        r.update_tags(ns='rio_overview', resampling=resampling.name)

            except (rasterio.errors.RasterioIOError, rasterio._err.CPLE_BaseError, KeyError) as e:
                raise GeoRaster2IOError(e)

    def __eq__(self, other):
        """Return True if GeoRasters are equal."""
        return self.crs == other.crs \
            and self.affine.almost_equals(other.affine) \
            and self.shape == other.shape \
            and self.image.dtype == other.image.dtype \
            and (
                (self.image.mask is np.ma.nomask or not np.any(self.image.mask)) and
                (other.image.mask is np.ma.nomask or not np.any(other.image.mask)) or
                np.array_equal(self.image.mask, other.image.mask)) \
            and np.array_equal(np.ma.filled(self.image, 0), np.ma.filled(other.image, 0))

    def __getitem__(self, key):
        """
        Crop raster by pixels.

        :param key: 2 slices: first over x, second over y.
        :return: new GeoRaster2
        """
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], slice):
            if key[0].step is not None or key[1].step is not None:
                raise GeoRaster2Error('Expected slicing with default step. Use raster.resize().')

            x_slice = key[0]
            y_slice = key[1]
            if (x_slice.start or 0) < 0 or (y_slice.start or 0) < 0:
                x_slice = slice(max(x_slice.start or 0, 0), x_slice.stop, x_slice.step)
                y_slice = slice(max(y_slice.start or 0, 0), y_slice.stop, y_slice.step)

                warnings.warn("Negative indices are not supported and were rounded to zero", GeoRaster2Warning)

            x_start = x_slice.start if x_slice.start is not None else 0
            y_start = y_slice.start if y_slice.start is not None else 0

            key = (slice(None, None, None), y_slice, x_slice)  # keep all bands

            affine_translated = self.affine * Affine.translation(x_start, y_start)

            return self.copy_with(image=self.image[key], affine=affine_translated)
        else:
            raise GeoRaster2Error('expected slice, got %s' % (key,))

    def get(self, point):
        """
        Get the pixel values at the requested point.

        :param point: A GeoVector(POINT) with the coordinates of the values to get
        :return: numpy array of values
        """
        if not (isinstance(point, GeoVector) and point.type == 'Point'):
            raise TypeError('expect GeoVector(Point), got %s' % (point,))

        target = self.to_raster(point)
        return self.image[:, int(target.y), int(target.x)]

    def __contains__(self, geometry):
        """
        Evaluate wether the geometry is fully contained in this raster

        :param geometry: A GeoVector to check
        :return: boolean
        """
        return self.footprint().contains(geometry)

    def astype(self, dst_type, in_range='dtype', out_range='dtype', clip_negative=False):
        """ Returns copy of the raster, converted to desired type
        Supported types: uint8, uint16, uint32, int8, int16, int32, float16, float32, float64

        :param dst_type: desired type
        :param in_range: str or 2-tuple, default 'dtype':
            'image': use image min/max as the intensity range,
            'dtype': use min/max of the image's dtype as the intensity range,
            2-tuple: use explicit min/max intensities, it is possible to use
            'min' or 'max' as tuple values - in this case they will be
            replaced by min or max intensity of image respectively
        :param out_range: str or 2-tuple, default 'dtype':
            'dtype': use min/max of the image's dtype as the intensity range,
            2-tuple: use explicit min/max intensities
        :param clip_negative: boolean, if `True` - clip the negative range, default False
        :return: numpy array of values
        """

        def type_max(dtype):
            return np.iinfo(dtype).max

        def type_min(dtype):
            return np.iinfo(dtype).min

        if (
            in_range is None and out_range is not None or
            out_range is None and in_range is not None
        ):
            raise GeoRaster2Error("Both ranges should be specified or none of them.")

        src_type = self.image.dtype
        if not np.issubdtype(src_type, np.integer) and in_range == 'dtype':
            in_range = 'image'
            warnings.warn("Value 'dtype' of in_range parameter is supported only for integer type. "
                          "Instead 'image' will be used.", GeoRaster2Warning)

        if not np.issubdtype(dst_type, np.integer) and out_range == 'dtype':
            raise GeoRaster2Error("Value 'dtype' of out_range parameter is supported only for integer type.")

        if (
            dst_type == src_type and
            in_range == out_range == 'dtype'
        ):
            return self

        # streching or shrinking intensity levels is required
        if out_range is not None:
            if out_range == 'dtype':
                omax = type_max(dst_type)
                omin = type_min(dst_type)
                if clip_negative and omin < 0:
                    omin = 0
            else:
                omin, omax = out_range

            if in_range == 'dtype':
                imin = type_min(src_type)
                imax = type_max(src_type)
            elif in_range == 'image':
                imin = min(self.min())
                imax = max(self.max())
            else:
                imin, imax = in_range
                if imin == 'min':
                    imin = min(self.min())
                if imax == 'max':
                    imax = max(self.max())

            if imin == imax:
                conversion_gain = 0
            else:
                conversion_gain = (omax - omin) / (imax - imin)

            # temp conversion, to handle saturation
            dst_array = conversion_gain * (self.image.astype(np.float) - imin) + omin
            dst_array = np.clip(dst_array, omin, omax)
        else:
            dst_array = self.image
        dst_array = dst_array.astype(dst_type)
        return self.copy_with(image=dst_array)

    def crop(self, vector, resolution=None, masked=True, bands=None):
        """
        crops raster outside vector (convex hull)
        :param vector: GeoVector
        :param resolution: output resolution, None for full resolution
        :return: GeoRaster
        """
        bounds, window = self._vector_to_raster_bounds(vector, boundless=self._image is None)
        if resolution:
            xsize, ysize = self._resolution_to_output_shape(bounds, resolution)
        else:
            xsize, ysize = (None, None)

        return self.pixel_crop(bounds, xsize, ysize, window=window, masked=masked, bands=bands)

    def _window(self, bounds, to_round=True):
        # self.window expects to receive the arguments west, south, east, north,
        # so for positive e in affine we should swap top and bottom
        if self.affine[4] > 0:
            window = self.window(bounds[0], bounds[3], bounds[2], bounds[1], precision=6)
        else:
            window = self.window(*bounds, precision=6)
        if to_round:
            window = window.round_offsets().round_shape(op='ceil', pixel_precision=3)
        return window

    def _vector_to_raster_bounds(self, vector, boundless=False):
        # bounds = tuple(round(bb) for bb in self.to_raster(vector).bounds)
        vector_bounds = vector.get_bounds(self.crs)
        if any(map(math.isinf, vector_bounds)):
            raise GeoRaster2Error('bounds %s cannot be transformed from %s to %s' % (
                vector.get_shape(vector.crs).bounds, vector.crs, self.crs))
        window = self._window(vector_bounds)
        (ymin, ymax), (xmin, xmax) = window.toranges()
        bounds = (xmin, ymin, xmax, ymax)
        if not boundless:
            left = max(0, bounds[0])
            bottom = max(0, bounds[1])
            right = min(self.width, bounds[2])
            top = min(self.height, bounds[3])
            bounds = (left, bottom, right, top)
        else:
            left, bottom, right, top = bounds

        width = right - left
        height = top - bottom
        window = rasterio.windows.Window(col_off=left, row_off=bottom, width=width, height=height)

        return bounds, window

    def _resolution_to_output_shape(self, bounds, resolution):
        base_resolution = abs(self.affine[0])
        if isinstance(resolution, (int, float)):
            xscale = yscale = resolution / base_resolution
        else:
            xscale = resolution[0] / base_resolution
            yscale = resolution[1] / base_resolution
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        xsize = round(width / xscale)
        ysize = round(height / yscale)
        return xsize, ysize

    def pixel_crop(self, bounds, xsize=None, ysize=None, window=None, masked=True, bands=None):
        """Crop raster outside vector (convex hull).

        :param bounds: bounds of requester portion of the image in image pixels
        :param xsize: output raster width, None for full resolution
        :param ysize: output raster height, None for full resolution
        :param windows: the bounds representation window on image in image pixels, Optional
        :param bands: list of indices of requested bands, default None which returns all bands
        :return: GeoRaster
        """

        if self._image is not None:
            raster = self._crop(bounds, xsize=xsize, ysize=ysize)
            if bands is not None:
                raster = raster.limit_to_bands(bands)
            return raster
        else:
            window = window or rasterio.windows.Window(bounds[0],
                                                       bounds[1],
                                                       bounds[2] - bounds[0] + 1,
                                                       bounds[3] - bounds[1] + 1)
            return self.get_window(window, xsize=xsize, ysize=ysize, masked=masked, bands=bands)

    def _crop(self, bounds, xsize=None, ysize=None):
        """Crop raster outside vector (convex hull).

        :param bounds: bounds on image
        :param xsize: output raster width, None for full resolution
        :param ysize: output raster height, None for full resolution
        :return: GeoRaster2
        """
        out_raster = self[
            int(bounds[0]): int(bounds[2]),
            int(bounds[1]): int(bounds[3])
        ]

        if xsize and ysize:
            if not (xsize == out_raster.width and ysize == out_raster.height):
                out_raster = out_raster.resize(dest_width=xsize, dest_height=ysize)
        return out_raster

    def attributes(self, url):
        """Without opening image, return size/bitness/bands/geography/...."""
        raise NotImplementedError

    def copy_with(self, **kwargs):
        """Get a copy of this GeoRaster with some attributes changed. NOTE: image is shallow-copied!"""
        init_args = {'affine': self.affine, 'crs': self.crs, 'band_names': self.band_names}
        init_args.update(kwargs)

        # The image is a special case because we don't want to make a copy of a possibly big array
        # unless totally necessary
        if 'image' not in init_args:
            init_args['image'] = self.image

        return self.__class__(**init_args)

    deepcopy_with = copy_with

    def __copy__(self):
        return self.copy_with()

    def __deepcopy__(self, memo):
        return self.deepcopy_with()

    def resolution(self):
        """Return resolution. if different in different axis - return geometric mean."""
        return resolution_from_affine(self.affine)

    def res_xy(self):
        """Returns X and Y resolution."""
        return abs(self.affine[0]), abs(self.affine[4])

    def resize(self, ratio=None, ratio_x=None, ratio_y=None, dest_width=None, dest_height=None, dest_resolution=None,
               resampling=Resampling.cubic):
        """
        Provide either ratio, or ratio_x and ratio_y, or dest_width and/or dest_height.

        :return: GeoRaster2
        """
        # validate input:
        if sum([ratio is not None, ratio_x is not None and ratio_y is not None,
                dest_height is not None or dest_width is not None, dest_resolution is not None]) != 1:
            raise GeoRaster2Error(
                'please provide either ratio, or {ratio_x,ratio_y}, or {dest_height, dest_width}, or dest_resolution')

        if dest_width is not None and dest_height is not None:
            ratio_x = float(dest_width) / self.width
            ratio_y = float(dest_height) / self.height
        elif dest_width is not None:
            ratio = float(dest_width) / self.width
        elif dest_height is not None:
            ratio = float(dest_height) / self.height
        elif dest_resolution is not None:
            ratio = self.resolution() / dest_resolution

        if ratio is not None:
            ratio_x, ratio_y = ratio, ratio

        return self._resize(ratio_x, ratio_y, resampling)

    def _resize(self, ratio_x, ratio_y, resampling):
        """Return raster resized by ratio."""
        new_width = int(np.ceil(self.width * ratio_x))
        new_height = int(np.ceil(self.height * ratio_y))
        dest_affine = self.affine * Affine.scale(1 / ratio_x, 1 / ratio_y)
        return self._reproject(new_width, new_height, dest_affine, resampling=resampling)

    def to_pillow_image(self, return_mask=False):
        """Return Pillow. Image, and optionally also mask."""
        img = np.rollaxis(np.rollaxis(self.image.data, 2), 2)
        img = Image.fromarray(img[:, :, 0]) if img.shape[2] == 1 else Image.fromarray(img)
        if return_mask:
            mask = np.ma.getmaskarray(self.image)
            mask = Image.fromarray(np.rollaxis(np.rollaxis(mask, 2), 2).astype(np.uint8)[:, :, 0])
            return img, mask
        else:
            return img

    @staticmethod
    def _patch_affine(affine):
        eps = 1e-100
        if (np.abs(affine) == np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1])).all():
            affine = affine * Affine.translation(eps, eps)
        return affine

    def _reproject(self, new_width, new_height, dest_affine, dtype=None,
                   dst_crs=None, resampling=Resampling.cubic):
        """Return re-projected raster to new raster.

        :param new_width: new raster width in pixels
        :param new_height: new raster height in pixels
        :param dest_affine: new raster affine
        :param dtype: new raster dtype, default current dtype
        :param dst_crs: new raster crs, default current crs
        :param resampling: reprojection resampling method, default `cubic`

        :return GeoRaster2
        """
        if new_width == 0 or new_height == 0:
            return None
        dst_crs = dst_crs or self.crs
        dtype = dtype or self.image.data.dtype
        dest_image = np.ma.masked_array(
            data=np.empty([self.num_bands, new_height, new_width], dtype=np.float32),
            mask=np.empty([self.num_bands, new_height, new_width], dtype=bool)
        )

        src_transform = self._patch_affine(self.affine)
        dst_transform = self._patch_affine(dest_affine)
        # first, reproject only data:
        rasterio.warp.reproject(self.image.data, dest_image.data, src_transform=src_transform,
                                dst_transform=dst_transform, src_crs=self.crs, dst_crs=dst_crs,
                                resampling=resampling)

        # rasterio.reproject has a bug for dtype=bool.
        # to bypass, manually convert mask to uint8, reproject, and convert back to bool:
        temp_mask = np.empty([self.num_bands, new_height, new_width], dtype=np.uint8)

        # extract the mask, and un-shrink if necessary
        mask = np.ma.getmaskarray(self.image)

        # rasterio.warp.reproject fills empty space with zeroes, which is the opposite of what
        # we want. therefore, we invert the mask so 0 is masked and 1 is unmasked, and we later
        # undo the inversion
        rasterio.warp.reproject((~mask).astype(np.uint8), temp_mask,
                                src_transform=src_transform, dst_transform=dst_transform,
                                src_crs=self.crs, dst_crs=dst_crs, resampling=Resampling.nearest)
        dest_image = np.ma.masked_array(dest_image.data.astype(dtype), temp_mask != 1)

        new_raster = self.copy_with(image=dest_image, affine=dst_transform, crs=dst_crs)

        return new_raster

    def reproject(self, dst_crs=None, resolution=None, dimensions=None,
                  src_bounds=None, dst_bounds=None, target_aligned_pixels=False,
                  resampling=Resampling.cubic, creation_options=None, **kwargs):
        """Return re-projected raster to new raster.

        Parameters
        ------------
        dst_crs: rasterio.crs.CRS, optional
            Target coordinate reference system.
        resolution: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target coordinate reference
            system.
        dimensions: tuple (width, height), optional
            Output size in pixels and lines.
        src_bounds: tuple (xmin, ymin, xmax, ymax), optional
            Georeferenced extent of output (in source georeferenced units).
        dst_bounds: tuple (xmin, ymin, xmax, ymax), optional
            Georeferenced extent of output (in destination georeferenced units).
        target_aligned_pixels: bool, optional
            Align the output bounds based on the resolution.
            Default is `False`.
        resampling: rasterio.enums.Resampling
            Reprojection resampling method. Default is `cubic`.
        creation_options: dict, optional
            Custom creation options.
        kwargs: optional
            Additional arguments passed to transformation function.

        Returns
        ---------
        out: GeoRaster2
        """
        if self._image is None and self._filename is not None:
            # image is not loaded yet
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tf:
                warp(self._filename, tf.name, dst_crs=dst_crs, resolution=resolution,
                     dimensions=dimensions, creation_options=creation_options,
                     src_bounds=src_bounds, dst_bounds=dst_bounds,
                     target_aligned_pixels=target_aligned_pixels,
                     resampling=resampling, **kwargs)

            new_raster = self.__class__(filename=tf.name, temporary=True,
                                        band_names=self.band_names)
        else:
            # image is loaded already
            # SimpleNamespace is handy to hold the properties that calc_transform expects, see
            # https://docs.python.org/3/library/types.html#types.SimpleNamespace
            src = SimpleNamespace(width=self.width, height=self.height, transform=self.transform, crs=self.crs,
                                  bounds=BoundingBox(*self.footprint().get_bounds(self.crs)),
                                  gcps=None)
            dst_crs, dst_transform, dst_width, dst_height = calc_transform(
                src, dst_crs=dst_crs, resolution=resolution, dimensions=dimensions,
                target_aligned_pixels=target_aligned_pixels,
                src_bounds=src_bounds, dst_bounds=dst_bounds)
            new_raster = self._reproject(dst_width, dst_height, dst_transform,
                                         dst_crs=dst_crs, resampling=resampling)

        return new_raster

    def to_png(self, transparent=True, thumbnail_size=None, resampling=None, in_range='dtype', out_range='dtype'):
        """
        Convert to png format (discarding geo).

        Optionally also resizes.
        Note: for color images returns interlaced.
        :param transparent: if True - sets alpha channel for nodata pixels
        :param thumbnail_size: if not None - resize to thumbnail size, e.g. 512
        :param in_range: input intensity range
        :param out_range: output intensity range
        :param resampling: one of Resampling enums

        :return bytes
        """
        return self.to_bytes(transparent=transparent, thumbnail_size=thumbnail_size,
                             resampling=resampling, in_range=in_range, out_range=out_range)

    def to_bytes(self, transparent=True, thumbnail_size=None, resampling=None, in_range='dtype', out_range='dtype',
                 format="png"):
        """
        Convert to selected format (discarding geo).

        Optionally also resizes.
        Note: for color images returns interlaced.
        :param transparent: if True - sets alpha channel for nodata pixels
        :param thumbnail_size: if not None - resize to thumbnail size, e.g. 512
        :param in_range: input intensity range
        :param out_range: output intensity range
        :param format : str, image format, default "png"
        :param resampling: one of Resampling enums

        :return bytes
        """
        resampling = resampling if resampling is not None else Resampling.cubic

        if self.num_bands < 3:
            warnings.warn("Deprecation: to_png of less then three bands raster will be not be supported in next \
release, please use: .colorize('gray').to_png()", GeoRaster2Warning)

        if self.num_bands > 3:
            warnings.warn("Limiting %d bands raster to first three bands to generate png" % self.num_bands,
                          GeoRaster2Warning)
            three_first_bands = self.band_names[:3]
            raster = self.limit_to_bands(three_first_bands)
        elif self.num_bands == 2:
            warnings.warn("Limiting two bands raster to use the first band to generate png",
                          GeoRaster2Warning)
            first_band = self.band_names[:1]
            raster = self.limit_to_bands(first_band)
        else:
            raster = self

        if raster.image.dtype != np.uint8:
            warnings.warn("downscaling dtype to 'uint8' to convert to png",
                          GeoRaster2Warning)
            thumbnail = raster.astype(np.uint8, in_range=in_range, out_range=out_range)
        else:
            thumbnail = raster.copy_with()

        if thumbnail_size:
            if thumbnail.width > thumbnail.height:
                thumbnail = thumbnail.resize(dest_width=thumbnail_size, resampling=resampling)
            else:
                thumbnail = thumbnail.resize(dest_height=thumbnail_size, resampling=resampling)

        img, mask = thumbnail.to_pillow_image(return_mask=True)

        if transparent:
            mask = np.array(mask)[:, :, np.newaxis]
            mask = 255 - 255 * mask  # inverse

            if thumbnail.num_bands == 1:
                img = np.stack([img, img, img], axis=2)  # make grayscale into rgb. bypass, as mode=LA isn't supported

            img = np.stack(tuple(np.split(np.asarray(img), 3, axis=2) + [mask]), axis=2)  # re-arrange into RGBA
            img = img[:, :, :, 0]

        f = io.BytesIO()
        imageio.imwrite(f, img, format)
        image_data = f.getvalue()
        return image_data

    @classmethod
    def from_bytes(cls, image_bytes, affine, crs, band_names=None):
        """Create GeoRaster from image BytesIo object.

        :param image_bytes: io.BytesIO object
        :param affine: rasters affine
        :param crs: rasters crs
        :param band_names: e.g. ['red', 'blue'] or 'red'
        """
        b = io.BytesIO(image_bytes)
        image = imageio.imread(b)
        roll = np.rollaxis(image, 2)
        if band_names is None:
            band_names = [0, 1, 2]
        elif isinstance(band_names, str):
            band_names = [band_names]

        return GeoRaster2(image=roll[:3, :, :], affine=affine, crs=crs, band_names=band_names)

    def _repr_png_(self):
        """Required for jupyter notebook to show raster."""
        return self.to_png(transparent=True, thumbnail_size=512, resampling=Resampling.nearest, in_range='image')

    def limit_to_bands(self, bands):
        bands_data = self.bands_data(bands)
        return self.copy_with(image=bands_data, band_names=bands)

    def num_pixels(self):
        return self.width * self.height

    def num_pixels_nodata(self):
        return np.sum(np.ma.getmaskarray(self.image)[0, :, :])

    def num_pixels_data(self):
        return np.sum(~np.ma.getmaskarray(self.image)[0, :, :])

    @classmethod
    def corner_types(cls):
        return ['ul', 'ur', 'br', 'bl']

    def image_corner(self, corner):
        """Return image corner in pixels, as shapely.Point."""
        if corner not in self.corner_types():
            raise GeoRaster2Error('corner %s invalid, expected: %s' % (corner, self.corner_types()))

        x = 0 if corner[1] == 'l' else self.width
        y = 0 if corner[0] == 'u' else self.height
        return Point(x, y)

    def corner(self, corner):
        """Return footprint origin in world coordinates, as GeoVector."""
        return self.to_world(self.image_corner(corner))

    def corners(self):
        """Return footprint corners, as {corner_type -> GeoVector}."""
        return {corner: self.corner(corner) for corner in self.corner_types()}

    def origin(self):
        """Return footprint origin in world coordinates, as GeoVector."""
        return self.corner('ul')

    def center(self):
        """Return footprint center in world coordinates, as GeoVector."""
        image_center = Point(self.width / 2, self.height / 2)
        return self.to_world(image_center)

    def bounds(self):
        """Return image rectangle in pixels, as shapely.Polygon."""
        corners = [self.image_corner(corner) for corner in self.corner_types()]
        return Polygon([[corner.x, corner.y] for corner in corners])

    def footprint(self):
        """Return rectangle in world coordinates, as GeoVector."""
        if self._footprint is not None:
            return self._footprint
        corners = [self.corner(corner) for corner in self.corner_types()]
        coords = []
        for corner in corners:
            shape = corner.get_shape(corner.crs)
            coords.append([shape.x, shape.y])

        shp = Polygon(coords)
        #  TODO use GeoVector.from_bounds
        self._footprint = GeoVector(shp, self.crs)
        return self._footprint

    def area(self):
        return self.footprint().area

    #  geography:
    def project(self, dst_crs, resampling):
        """Return reprojected raster."""

    def to_raster(self, vector):
        """Return the vector in pixel coordinates, as shapely.Geometry."""
        return transform(vector.get_shape(vector.crs), vector.crs, self.crs, dst_affine=~self.affine)

    def to_world(self, shape, dst_crs=None):
        """Return the shape (provided in pixel coordinates) in world coordinates, as GeoVector."""
        if dst_crs is None:
            dst_crs = self.crs
        shp = transform(shape, self.crs, dst_crs, dst_affine=self.affine)
        return GeoVector(shp, dst_crs)

    #  array:
    # array ops: bitness conversion, setitem/getitem slices, +-*/.. scalar
    def min(self):
        return self.reduce('min')

    def max(self):
        return self.reduce('max')

    def sum(self):
        return self.reduce('sum')

    def mean(self):
        return self.reduce('mean')

    def var(self):
        return self.reduce('var')

    def std(self):
        return self.reduce('std')

    def reduce(self, op):
        """Reduce the raster to a score, using 'op' operation.

        nodata pixels are ignored.
        op is currently limited to numpy.ma, e.g. 'mean', 'std' etc
        :returns list of per-band values
        """
        per_band = [getattr(np.ma, op)(self.image.data[band, np.ma.getmaskarray(self.image)[band, :, :] == np.False_])
                    for band in range(self.num_bands)]
        return per_band

    def histogram(self):
        histogram = {band: self._histogram(band) for band in self.band_names}
        return Histogram(histogram)

    def _histogram(self, band):
        if self.image.dtype == np.uint8:
            length = 256
        elif self.image.dtype == np.uint16:
            length = 65536
        else:
            raise GeoRaster2NotImplementedError('cant calculate histogram for type %s' % self.image.dtype)

        band_image = self.limit_to_bands(band).image
        return np.histogram(band_image[~np.ma.getmaskarray(band_image)], range=(0, length), bins=length)[0]

    #  image:
    # wrap, if required, some image processing functions, to deal with nodata pixels.

    #  image + geography:
    def apply_transform(self, transformation, resampling):
        """
        Apply affine transformation on image & georeferencing.

        as specific cases, implement 'resize', 'rotate', 'translate'
        """
        raise NotImplementedError

    def rectify(self):
        """Rotate raster northwards."""
        raise NotImplementedError

    #  vs. GeoVector:

    def vectorize(self, condition=None):
        """
        Return GeoVector of raster, excluding nodata pixels, subject to 'condition'.

        :param condition: e.g. 42 < value < 142.

        e.g. if no nodata pixels, and without condition - this == footprint().
        """
        raise NotImplementedError

    def __invert__(self):
        """Invert mask."""
        return self.copy_with(image=np.ma.masked_array(self.image.data, np.logical_not(np.ma.getmaskarray(self.image))))

    def mask(self, vector, mask_shape_nodata=False):
        """
        Set pixels outside vector as nodata.

        :param vector: GeoVector, GeoFeature, FeatureCollection
        :param mask_shape_nodata: if True - pixels inside shape are set nodata, if False - outside shape is nodata
        :return: GeoRaster2
        """
        # shape = vector.reproject(self.crs).shape
        if isinstance(vector, Iterable):
            shapes = [self.to_raster(feature) for feature in vector]
        else:
            shapes = [self.to_raster(vector)]

        mask = geometry_mask(shapes, (self.height, self.width), Affine.identity(), invert=mask_shape_nodata)
        masked = self.deepcopy_with()
        masked.image.mask |= mask
        return masked

    #  vs. GeoRaster:
    def add_raster(self, other, merge_strategy, resampling):
        """
        Return merge of 2 rasters, in geography of the first one.

        merge_strategy - for pixels with values in both rasters.
        """
        raise NotImplementedError

    def __add__(self, other):
        return self.add_raster(other, merge_strategy='prefer left operand', resampling='nearest')

    def merge(self, other, merge_strategy=MergeStrategy.UNION):
        # TODO: Evaluate whether this should be add_raster
        return merge_two(self, other, merge_strategy)

    def intersect(self, other):
        """Pixels outside either raster are set nodata"""
        raise NotImplementedError

    #  tiles:
    def to_tiles(self):
        """Yield slippy-map tiles."""
        raise NotImplementedError

    @classmethod
    def from_tiles(cls, tiles):
        """Compose raster from tiles. return GeoRaster."""
        raise NotImplementedError

    def _overviews_factors(self, blocksize=256):
        return _calc_overviews_factors(self, blocksize=blocksize)

    def save_cloud_optimized(self, dest_url, resampling=Resampling.gauss):
        """Save as Cloud Optimized GeoTiff object to a new file.

        :param dest_url: path to the new raster
        :param resampling: which Resampling to use on reading, default Resampling.gauss

        :return: new VirtualGeoRaster of the tiled object
        """

        src = self  # GeoRaster2.open(self._filename)

        with tempfile.NamedTemporaryFile(suffix='.tif') as tf:
            src.save(tf.name, overviews=False)
            convert_to_cog(tf.name, dest_url, resampling)

        geotiff = GeoRaster2.open(dest_url)
        return geotiff

    def _get_window_out_shape(self, bands, window, xsize, ysize):
        """Get the outshape of a window.

        this method is only used inside get_window to calculate the out_shape
        """

        if xsize and ysize is None:
            ratio = window.width / xsize
            ysize = math.ceil(window.height / ratio)
        elif ysize and xsize is None:
            ratio = window.height / ysize
            xsize = math.ceil(window.width / ratio)
        elif xsize is None and ysize is None:
            ysize = math.ceil(window.height)
            xsize = math.ceil(window.width)
        return (len(bands), ysize, xsize)

    def get_window(self, window, bands=None,
                   xsize=None, ysize=None,
                   resampling=Resampling.cubic, masked=True, affine=None
                   ):
        """Get window from raster.

        :param window: requested window
        :param bands: list of indices of requested bads, default None which returns all bands
        :param xsize: tile x size default None, for full resolution pass None
        :param ysize: tile y size default None, for full resolution pass None
        :param resampling: which Resampling to use on reading, default Resampling.cubic
        :param masked: boolean, if `True` the return value will be a masked array. Default is True
        :return: GeoRaster2 of tile
        """
        bands = bands or list(range(1, self.num_bands + 1))

        # requested_out_shape and out_shape are different for out of bounds window
        out_shape = self._get_window_out_shape(bands, window, xsize, ysize)
        try:
            read_params = {
                "window": window,
                "resampling": resampling,
                "boundless": True,
                "masked": masked,
                "out_shape": out_shape
            }
            with self._raster_opener(self._filename) as raster:  # type: rasterio.io.DatasetReader
                array = raster.read(bands, **read_params)
            affine = affine or self._calculate_new_affine(window, out_shape[2], out_shape[1])
            raster = self.copy_with(image=array, affine=affine)

            if masked and not raster.image.mask.any():
                intersection = self.footprint().envelope.intersection(raster.footprint().envelope)
                if not intersection.is_empty:
                    raster = raster.mask(intersection)
                else:
                    raster.image.mask = True

            return raster

        except (rasterio.errors.RasterioIOError, rasterio._err.CPLE_HttpResponseError) as e:
            raise GeoRaster2IOError(e)

    def _get_tile_when_web_mercator_crs(self, x_tile, y_tile, zoom,
                                        bands=None, masked=False,
                                        resampling=Resampling.cubic):
        """ The reason we want to treat this case in a special way
            is that there are cases where the rater is aligned so you need to be precise
            on which raster you want
        """
        roi = GeoVector.from_xyz(x_tile, y_tile, zoom)
        coordinates = roi.get_bounds(WEB_MERCATOR_CRS)
        window = self._window(coordinates, to_round=False)
        bands = bands or list(range(1, self.num_bands + 1))
        # we know the affine the result should produce becuase we know where
        # it is located by the xyz, therefore we calculate it here
        ratio = MERCATOR_RESOLUTION_MAPPING[zoom] / self.resolution()

        # the affine should be calculated before rounding the window values
        affine = self.window_transform(window)
        affine = affine * Affine.scale(ratio, ratio)

        window = Window(round(window.col_off),
                        round(window.row_off),
                        round(window.width),
                        round(window.height))
        return self.get_window(window, bands=bands, xsize=256, ysize=256, masked=masked, affine=affine)

    def get_tile(self, x_tile, y_tile, zoom,
                 bands=None, masked=False, resampling=Resampling.cubic):
        """Convert mercator tile to raster window.

        :param x_tile: x coordinate of tile
        :param y_tile: y coordinate of tile
        :param zoom: zoom level
        :param bands: list of indices of requested bands, default None which returns all bands
        :return: GeoRaster2 of tile in WEB_MERCATOR_CRS

        You can use TELLURIC_GET_TILE_BUFFER env variable to control the number of pixels surrounding
        the vector you should fetch when using this method on a raster that is not in WEB_MERCATOR_CRS
        default to 10
        """
        if self.crs == WEB_MERCATOR_CRS:
            return self._get_tile_when_web_mercator_crs(x_tile, y_tile, zoom, bands, masked, resampling)

        roi = GeoVector.from_xyz(x_tile, y_tile, zoom)
        left, bottom, right, top = roi.get_bounds(WEB_MERCATOR_CRS)
        new_affine = rasterio.warp.calculate_default_transform(WEB_MERCATOR_CRS, self.crs,
                                                               256, 256, left, bottom, right, top)[0]
        new_resolution = resolution_from_affine(new_affine)
        buffer_ratio = int(os.environ.get("TELLURIC_GET_TILE_BUFFER", 10))
        roi_buffer = roi.buffer(math.sqrt(roi.area * buffer_ratio / 100))
        raster = self.crop(roi_buffer, resolution=new_resolution, masked=masked, bands=bands)
        raster = raster.reproject(dst_crs=WEB_MERCATOR_CRS, resolution=MERCATOR_RESOLUTION_MAPPING[zoom],
                                  dst_bounds=roi_buffer.get_bounds(WEB_MERCATOR_CRS),
                                  resampling=Resampling.cubic_spline)
        # raster = raster.get_tile(x_tile, y_tile, zoom, bands, masked, resampling)
        raster = raster.crop(roi).resize(dest_width=256, dest_height=256)
        return raster

    def _calculate_new_affine(self, window, blockxsize=256, blockysize=256):
        new_affine = self.window_transform(window)
        width = math.ceil(abs(window.width))
        height = math.ceil(abs(window.height))
        x_scale = width / blockxsize
        y_scale = height / blockysize
        new_affine = new_affine * Affine.scale(x_scale, y_scale)
        return new_affine

    def colorize(self, colormap, band_name=None, vmin=None, vmax=None):
        """Apply a colormap on a selected band.

        colormap list: https://matplotlib.org/examples/color/colormaps_reference.html

        Parameters
        ----------
        colormap : str
        Colormap name from this list https://matplotlib.org/examples/color/colormaps_reference.html

        band_name : str, optional
        Name of band to colorize, if none the first band will be used

        vmin, vmax : int, optional
        minimum and maximum range for normalizing array values, if None actual raster values will be used

        Returns
        -------
        GeoRaster2
        """
        vmin = vmin if vmin is not None else min(self.min())
        vmax = vmax if vmax is not None else max(self.max())

        cmap = plt.get_cmap(colormap)  # type: matplotlib.colors.Colormap

        band_index = 0
        if band_name is None:
            if self.num_bands > 1:
                warnings.warn("Using the first band to colorize the raster", GeoRaster2Warning)
        else:
            band_index = self.band_names.index(band_name)

        normalized = (self.image[band_index, :, :] - vmin) / (vmax - vmin)

        # Colormap instances are used to convert data values (floats)
        # to RGBA color that the respective Colormap
        #
        # https://matplotlib.org/_modules/matplotlib/colors.html#Colormap
        image_data = cmap(normalized)
        image_data = image_data[:, :, 0:3]

        # convert floats [0,1] to uint8 [0,255]
        image_data = image_data * 255
        image_data = image_data.astype(np.uint8)

        image_data = np.rollaxis(image_data, 2)

        # force nodata where it was in original raster:
        mask = _join_masks_from_masked_array(self.image)
        mask = np.stack([mask[0, :, :]] * 3)
        array = np.ma.array(image_data.data, mask=mask).filled(0)  # type: np.ndarray
        array = np.ma.array(array, mask=mask)

        return self.copy_with(image=array, band_names=['red', 'green', 'blue'])


class Histogram:
    def __init__(self, hist=None):
        """
        :param hist: {band -> ndarray}
        """
        self.hist = hist

    @property
    def length(self):
        """Return number of bins in single-band histogram."""
        return len(next(iter(self.hist.values())))

    @property
    def bins(self):
        return range(self.length)

    def __str__(self):
        return self.hist.__str__()

    def __getitem__(self, band):
        """Return histogram of single band, as ndarray."""
        return self.hist[band]

    def _repr_png_(self):
        plt.figure(figsize=(18, 6))
        plt.title('histogram')
        plt.xlabel('intensity')
        plt.ylabel('frequency')
        plt.xlim(0, self.length - 1)
        for band in self.hist:
            plt.plot(self.bins, self.hist[band], label=band)
        plt.legend(loc='upper left')
