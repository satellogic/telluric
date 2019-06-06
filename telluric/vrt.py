"""borrowed from Rasterio"""
import os
import json
import xml.etree.ElementTree as ET

from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.windows import from_bounds, Window
from telluric.base_vrt import BaseVRT

from typing import Dict, List, Any


def find_and_convert_to_type(_type, node, path, default=None):
    value = node.find(path)
    if value is not None:
        value = _type(value.text)
    else:
        value = default
    return value


def wms_vrt(wms_file, bounds=None, resolution=None):
    """Make a VRT XML document from a wms file.
    Parameters
    ----------
    wms_file : str
        The source wms file
    bounds : GeoVector, optional
        The requested footprint of the generated VRT
    resolution : float, optional
        The requested resolution of the generated VRT
    Returns
    -------
    bytes
        An ascii-encoded string (an ElementTree detail)
    """

    from telluric import rasterization, constants
    wms_tree = ET.parse(wms_file)
    service = wms_tree.find(".//Service")
    if service is not None:
        service_name = service.attrib.get("name")
    else:
        raise ValueError("Service tag is required")
    # definition is based on https://www.gdal.org/frmt_wms.html
    if service_name == "VirtualEarth":
        left = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftX", -20037508.34)
        up = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftY", 20037508.34)
        right = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightX", 20037508.34)
        bottom = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightY", -20037508.34)
        upper_bound_zoom = find_and_convert_to_type(int, wms_tree, ".//DataWindow/TileLevel", 19)
        projection = find_and_convert_to_type(str, wms_tree, ".//Projection", "EPSG: 3857")
        projection = CRS(init=projection)
        blockx = find_and_convert_to_type(str, wms_tree, ".//BlockSizeX", 256)
        blocky = find_and_convert_to_type(str, wms_tree, ".//BlockSizeY", 256)
    else:
        left = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftX", -180.0)
        up = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftY", 90.0)
        right = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightX", 180.0)
        bottom = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightY", -90.0)
        upper_bound_zoom = find_and_convert_to_type(int, wms_tree, ".//DataWindow/TileLevel", 0)
        projection = find_and_convert_to_type(str, wms_tree, ".//Projection", "EPSG:4326")
        blockx = find_and_convert_to_type(str, wms_tree, ".//BlockSizeX", 1024)
        blocky = find_and_convert_to_type(str, wms_tree, ".//BlockSizeY", 1024)
        projection = CRS(init=projection)

    bands_count = find_and_convert_to_type(int, wms_tree, ".//BandsCount", 3)
    data_type = find_and_convert_to_type(str, wms_tree, ".//DataType", "Byte")

    src_bounds = (left, bottom, right, up)
    bounds = bounds.get_bounds(crs=projection) or src_bounds
    src_resolution = constants.MERCATOR_RESOLUTION_MAPPING[upper_bound_zoom]
    resolution = resolution or constants.MERCATOR_RESOLUTION_MAPPING[upper_bound_zoom]
    dst_width, dst_height, transform = rasterization.raster_data(bounds=bounds, dest_resolution=resolution)
    orig_width, orig_height, orig_transform = rasterization.raster_data(
        bounds=src_bounds, dest_resolution=src_resolution)
    src_window = from_bounds(*bounds, transform=orig_transform)

    vrt = BaseVRT(dst_width, dst_height, projection, transform)

    vrt.add_metadata(domain="IMAGE_STRUCTURE", items={"INTERLEAVE": "PIXEL"})

    if bands_count != 3:
        raise ValueError("We support currently on 3 bands WMS")

    for idx, band in enumerate(["RED", "GREEN", "BLUE"]):
        bidx = idx + 1

        band_element = vrt.add_band(data_type, bidx, band)
        dst_window = Window(0, 0, dst_width, dst_height)

        vrt.add_band_simplesource(band_element, bidx, data_type, False, os.path.abspath(wms_file),
                                  orig_width, orig_height, blockx, blocky,
                                  src_window, dst_window)

    return vrt


def boundless_vrt_doc(
        src_dataset, nodata=None, background=None, hidenodata=False,
        width=None, height=None, transform=None, bands=None):
    """Make a VRT XML document.
    Parameters
    ----------
    src_dataset : Dataset
        The dataset to wrap.
    background : Dataset, optional
        A dataset that provides the optional VRT background. NB: this dataset
        must have the same number of bands as the src_dataset.
    Returns
    -------
    bytes
        An ascii-encoded string (an ElementTree detail)
    """

    nodata = nodata or src_dataset.nodata
    width = width or src_dataset.width
    height = height or src_dataset.height
    transform = transform or src_dataset.transform

    vrt = BaseVRT(width, height, src_dataset.crs, transform)

    bidxs = src_dataset.indexes if bands is None else bands
    for bidx in bidxs:
        ci = src_dataset.colorinterp[bidx - 1]
        block_shape = src_dataset.block_shapes[bidx - 1]
        dtype = src_dataset.dtypes[bidx - 1]

        band_element = vrt.add_band(dtype, bidx, ci.name, nodata=nodata, hidenodata=True)

        if background is not None:
            src_window = Window(0, 0, background.width, background.height)
            dst_window = Window(0, 0, width, height)
            vrt.add_band_simplesource(band_element, bidx, dtype, False, background.name,
                                      width, height, block_shape[1], block_shape[0],
                                      src_window, dst_window)

        src_window = Window(0, 0, src_dataset.width, src_dataset.height)
        xoff = (src_dataset.transform.xoff - transform.xoff) / transform.a
        yoff = (src_dataset.transform.yoff - transform.yoff) / transform.e
        xsize = src_dataset.width * src_dataset.transform.a / transform.a
        ysize = src_dataset.height * src_dataset.transform.e / transform.e
        dst_window = Window(xoff, yoff, xsize, ysize)
        vrt.add_band_simplesource(band_element, bidx, dtype, False, src_dataset.name,
                                  width, height, block_shape[1], block_shape[0],
                                  src_window, dst_window, nodata=src_dataset.nodata)

    if all(MaskFlags.per_dataset in flags for flags in src_dataset.mask_flag_enums):
        mask_band = vrt.add_mask_band('Byte')
        src_window = Window(0, 0, src_dataset.width, src_dataset.height)
        xoff = (src_dataset.transform.xoff - transform.xoff) / transform.a
        yoff = (src_dataset.transform.yoff - transform.yoff) / transform.e
        xsize = src_dataset.width
        ysize = src_dataset.height
        dst_window = Window(xoff, yoff, xsize, ysize)
        vrt.add_band_simplesource(mask_band, 'mask,1', 'Byte', False, src_dataset.name,
                                  width, height, block_shape[1], block_shape[0], src_window, dst_window)
    return vrt


def band_name_to_color_interpretation(band_name):
    if isinstance(band_name, str) and band_name.lower() in ['red', 'green', 'blue']:
        return band_name
    else:
        return 'Gray'


def raster_list_vrt(rasters, relative_to_vrt=True, nodata=None, mask_band=None):
    """Make a VRT XML document from a list of GeoRaster2 objects.
    Parameters
    ----------
    rasters : list
        The list of GeoRasters.
    relative_to_vrt : bool, optional
        If True the bands simple source url will be related to the VRT file
    nodata : int, optional
        If supplied is the note data value to be used
    mask_band: int, optional
        If specified detrimes from which band to use the mask
    Returns
    -------
    bytes
        An ascii-encoded string (an ElementTree detail)
    """

    from telluric import FeatureCollection
    fc = FeatureCollection.from_georasters(rasters)
    return raster_collection_vrt(fc, relative_to_vrt, nodata, mask_band)


def raster_collection_vrt(fc, relative_to_vrt=True, nodata=None, mask_band=None):
    """Make a VRT XML document from a feature collection of GeoRaster2 objects.
    Parameters
    ----------
    rasters : FeatureCollection
        The FeatureCollection of GeoRasters.
    relative_to_vrt : bool, optional
        If True the bands simple source url will be related to the VRT file
    nodata : int, optional
        If supplied is the note data value to be used
    mask_band: int, optional
        If specified detrimes from which band to use the mask
    Returns
    -------
    bytes
        An ascii-encoded string (an ElementTree detail)
    """

    def max_resolution():
        max_affine = max(fc, key=lambda f: f.raster().resolution()).raster().affine
        return abs(max_affine.a), abs(max_affine.e)

    from telluric import rasterization
    from telluric.georaster import band_names_tag
    assert all(fc.crs == f.crs for f in fc), "all rasters should have the same CRS"

    rasters = (f.raster() for f in fc)
    bounds = fc.convex_hull.get_bounds(fc.crs)
    resolution = max_resolution()
    width, height, affine = rasterization.raster_data(bounds, resolution)

    bands = {}  # type: Dict[str, tuple]
    band_names = []  # type: List[Any]
    vrt = BaseVRT(width, height, fc.crs, affine)

    last_band_idx = 0
    if mask_band is not None:
        mask_band_elem = vrt.add_mask_band("Byte")

    for raster in rasters:
        for i, band_name in enumerate(raster.band_names):
            if band_name in bands:
                band_element, band_idx = bands[band_name]
            else:
                last_band_idx += 1
                band_idx = last_band_idx
                band_element = vrt.add_band(raster.dtype, band_idx, band_name_to_color_interpretation(band_name),
                                            nodata=nodata)
                bands[band_name] = (band_element, last_band_idx)
                band_names.append(band_name)

            src_window = Window(0, 0, raster.width, raster.height)
            xoff = (raster.affine.xoff - affine.xoff) / affine.a
            yoff = (raster.affine.yoff - affine.yoff) / affine.e
            xsize = raster.width * raster.affine.a / affine.a
            ysize = raster.height * raster.affine.e / affine.e
            dst_window = Window(xoff, yoff, xsize, ysize)
            if raster.source_file.startswith("http"):
                file_name = "/vsicurl/%s" % raster.source_file
            elif relative_to_vrt:
                file_name = raster.source_file
            else:
                file_name = os.path.join(os.getcwd(), raster.source_file)

            vrt.add_band_simplesource(band_element, band_idx, raster.dtype, relative_to_vrt, file_name,
                                      raster.width, raster.height,
                                      raster.block_shape(i)[1], raster.block_shape(i)[0],
                                      src_window, dst_window)
            if i == mask_band:
                vrt.add_band_simplesource(mask_band_elem, "mask,%s" % (mask_band + 1),
                                          "Byte",
                                          relative_to_vrt,
                                          file_name,
                                          raster.width, raster.height,
                                          raster.block_shape(i)[1], raster.block_shape(i)[0],
                                          src_window, dst_window)

    vrt.add_metadata(items={band_names_tag: json.dumps(band_names)})
    return vrt
