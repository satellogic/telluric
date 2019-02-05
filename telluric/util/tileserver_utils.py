import telluric as tl
import mercantile
from telluric.util.raster_utils import _calc_overviews_factors, warp, _get_telluric_tags
from affine import Affine
import time
import rasterio
from tempfile import TemporaryDirectory
import os
from rasterio import shutil as rasterio_sh


def mercator_upper_zoom_level(raster):
    resolution = raster.resolution()
    for zoom, mercartor_resolution in tl.constants.MERCATOR_RESOLUTION_MAPPING.items():
        if resolution > mercartor_resolution:
            return mercartor_resolution
    return None


def tileserver_optimized_raster(src, dest):
    """ This method converts a raster to a tileserver optimized raster.
        The method will reproject the raster to align to the xyz system, in resolution and projection
        It will also create overviews
        And finally it will arragne the raster in a cog way.
        You could take the dest file upload it to a web server that supports ranges and user GeoRaster.get_tile
        on it,
        You are geranteed that you will get as minimal data as possible
    """
    src_raster = tl.GeoRaster2.open(src)
    bounding_box = src_raster.footprint().get_shape(tl.constants.WGS84_CRS).bounds
    tile = mercantile.bounding_tile(*bounding_box)
    dest_resolution = mercator_upper_zoom_level(src_raster)
    bounds = tl.GeoVector.from_xyz(tile.x, tile.y, tile.z).get_bounds(tl.constants.WEB_MERCATOR_CRS)
    create_options = {
        "tiled": "YES",
        "blocksize": 256,
        "compress": "DEFLATE",
        "photometric": "MINISBLACK"
    }
    with TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'temp.tif')

        warp(src, temp_file, dst_crs=tl.constants.WEB_MERCATOR_CRS, resolution=dest_resolution,
             dst_bounds=bounds, create_options=create_options)

        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE=256):
            resampling = rasterio.enums.Resampling.gauss
            with rasterio.open(temp_file, 'r+') as tmp_raster:
                factors = _calc_overviews_factors(tmp_raster)
                tmp_raster.build_overviews(factors, resampling=resampling)
                tmp_raster.update_tags(ns='rio_overview', resampling=resampling.name)
                telluric_tags = _get_telluric_tags(src)
                if telluric_tags:
                    tmp_raster.update_tags(**telluric_tags)

            rasterio_sh.copy(temp_file, dest,
                             COPY_SRC_OVERVIEWS=True, tiled=True,
                             compress='DEFLATE', photometric='MINISBLACK')
