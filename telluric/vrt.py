"""borrowed from Rasterio"""

import xml.etree.ElementTree as ET

from rasterio.dtypes import _gdal_typename
from rasterio.enums import MaskFlags
from rasterio.path import parse_path, vsi_path
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
import os
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def find_and_convert_to_type(_type, node, path):
    value = node.find(path)
    if value is not None:
        value = _type(value.text)
    return value


def wms_vrt(wms_file, bounds=None, resolution=None):
    from telluric import rasterization, constants
    wms_tree = ET.parse(wms_file)
    left = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftX")
    up = find_and_convert_to_type(float, wms_tree, ".//DataWindow/UpperLeftY")
    right = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightX")
    bottom = find_and_convert_to_type(float, wms_tree, ".//DataWindow/LowerRightY")
    src_bounds = (left, bottom, right, up)
    bounds = bounds or src_bounds
    upper_bound_zoom = find_and_convert_to_type(int, wms_tree, ".//DataWindow/TileLevel")
    src_resolution = constants.MERCATOR_RESOLUTION_MAPPING[upper_bound_zoom]
    resolution = resolution or constants.MERCATOR_RESOLUTION_MAPPING[upper_bound_zoom]
    dst_height, dst_width, transform = rasterization.raster_data(bounds=bounds, dest_resolution=resolution)
    orig_height, orig_width, orig_transform = rasterization.raster_data(
        bounds=src_bounds, dest_resolution=src_resolution)
    src_window = from_bounds(*bounds, orig_transform)
    vrtdataset = ET.Element('VRTDataset')
    vrtdataset.attrib['rasterXSize'] = str(dst_height)
    vrtdataset.attrib['rasterYSize'] = str(dst_width)
    srs = ET.SubElement(vrtdataset, 'SRS')
    projection = find_and_convert_to_type(str, wms_tree, ".//Projection")
    blockx = find_and_convert_to_type(str, wms_tree, ".//BlockSizeX")
    blocky = find_and_convert_to_type(str, wms_tree, ".//BlockSizeY")
    projection = CRS(init=projection)
    srs.text = projection.wkt
    geotransform = ET.SubElement(vrtdataset, 'GeoTransform')
    geotransform.text = ','.join([str(v) for v in transform.to_gdal()])
    image_metadata = ET.SubElement(vrtdataset, 'Metadata')
    image_metadata.attrib["domain"] = "IMAGE_STRUCTURE"
    image_mdi = ET.SubElement(image_metadata, "MDI")
    image_mdi.attrib["key"] = "INTERLEAVE"
    image_mdi.text = "PIXEL"
    bands_count = find_and_convert_to_type(int, wms_tree, ".//BandsCount")
    if bands_count != 3:
        raise ValueError("We support corrently on 3 bands WMS")
    for idx, band in enumerate(["RED", "GREEN", "BLUE"]):
        bidx = idx + 1
        vrtrasterband = ET.SubElement(vrtdataset, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = "Byte"
        vrtrasterband.attrib['band'] = str(bidx)
        colorinterp = ET.SubElement(vrtrasterband, 'ColorInterp')
        colorinterp.text = band
        simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        sourcefilename = ET.SubElement(simplesource, "SourceFilename")
        sourcefilename.attrib["relativeToVRT"] = "0"
        sourcefilename.text = os.path.abspath(wms_file)
        sourceband = ET.SubElement(simplesource, "sourceband")
        sourceband.text = str(bidx)
        sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
        sourceproperties.attrib['RasterYSize'] = str(orig_width)
        sourceproperties.attrib['RasterXSize'] = str(orig_height)
        sourceproperties.attrib['DataType'] = "Byte"
        sourceproperties.attrib['BlockXSize'] = blockx
        sourceproperties.attrib['BlockYSize'] = blocky
        srcrect = ET.SubElement(simplesource, 'SrcRect')
        srcrect.attrib["yOff"] = str(src_window.row_off)
        srcrect.attrib["xOff"] = str(src_window.col_off)
        srcrect.attrib["ySize"] = str(src_window.height)
        srcrect.attrib["xSize"] = str(src_window.width)
        dstrect = ET.SubElement(simplesource, 'DstRect')
        dstrect.attrib["xOff"] = "0"
        dstrect.attrib["yOff"] = "0"
        dstrect.attrib["xSize"] = str(dst_height)
        dstrect.attrib["ySize"] = str(dst_width)
    return ET.tostring(vrtdataset)


def boundless_vrt_doc(
        src_dataset, nodata=None, background=None, hidenodata=False,
        width=None, height=None, transform=None):
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

    vrtdataset = ET.Element('VRTDataset')
    vrtdataset.attrib['rasterYSize'] = str(height)
    vrtdataset.attrib['rasterXSize'] = str(width)
    srs = ET.SubElement(vrtdataset, 'SRS')
    srs.text = src_dataset.crs.wkt if src_dataset.crs else ""
    geotransform = ET.SubElement(vrtdataset, 'GeoTransform')
    geotransform.text = ','.join([str(v) for v in transform.to_gdal()])

    for bidx, ci, block_shape, dtype in zip(src_dataset.indexes, src_dataset.colorinterp,
                                            src_dataset.block_shapes, src_dataset.dtypes):
        vrtrasterband = ET.SubElement(vrtdataset, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = _gdal_typename(dtype)
        vrtrasterband.attrib['band'] = str(bidx)

        if nodata is not None:
            nodatavalue = ET.SubElement(vrtrasterband, 'NoDataValue')
            nodatavalue.text = str(nodata)

            if hidenodata:
                hidenodatavalue = ET.SubElement(vrtrasterband, 'HideNoDataValue')
                hidenodatavalue.text = "1"

        colorinterp = ET.SubElement(vrtrasterband, 'ColorInterp')
        colorinterp.text = ci.name.capitalize()

        if background is not None:
            simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
            sourcefilename = ET.SubElement(simplesource, 'SourceFilename')
            sourcefilename.attrib['relativeToVRT'] = "0"
            sourcefilename.text = vsi_path(parse_path(background.name))
            sourceband = ET.SubElement(simplesource, 'SourceBand')
            sourceband.text = str(bidx)
            sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
            sourceproperties.attrib['RasterXSize'] = str(width)
            sourceproperties.attrib['RasterYSize'] = str(height)
            sourceproperties.attrib['dataType'] = _gdal_typename(dtype)
            sourceproperties.attrib['BlockYSize'] = str(block_shape[0])
            sourceproperties.attrib['BlockXSize'] = str(block_shape[1])
            srcrect = ET.SubElement(simplesource, 'SrcRect')
            srcrect.attrib['xOff'] = '0'
            srcrect.attrib['yOff'] = '0'
            srcrect.attrib['xSize'] = str(background.width)
            srcrect.attrib['ySize'] = str(background.height)
            dstrect = ET.SubElement(simplesource, 'DstRect')
            dstrect.attrib['xOff'] = '0'
            dstrect.attrib['yOff'] = '0'
            dstrect.attrib['xSize'] = str(width)
            dstrect.attrib['ySize'] = str(height)

        simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        sourcefilename = ET.SubElement(simplesource, 'SourceFilename')
        sourcefilename.attrib['relativeToVRT'] = "0"
        sourcefilename.text = vsi_path(parse_path(src_dataset.name))
        sourceband = ET.SubElement(simplesource, 'SourceBand')
        sourceband.text = str(bidx)
        sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(width)
        sourceproperties.attrib['RasterYSize'] = str(height)
        sourceproperties.attrib['dataType'] = _gdal_typename(dtype)
        sourceproperties.attrib['BlockYSize'] = str(block_shape[0])
        sourceproperties.attrib['BlockXSize'] = str(block_shape[1])
        srcrect = ET.SubElement(simplesource, 'SrcRect')
        srcrect.attrib['xOff'] = '0'
        srcrect.attrib['yOff'] = '0'
        srcrect.attrib['xSize'] = str(src_dataset.width)
        srcrect.attrib['ySize'] = str(src_dataset.height)
        dstrect = ET.SubElement(simplesource, 'DstRect')
        dstrect.attrib['xOff'] = str((src_dataset.transform.xoff - transform.xoff) / transform.a)
        dstrect.attrib['yOff'] = str((src_dataset.transform.yoff - transform.yoff) / transform.e)
        dstrect.attrib['xSize'] = str(src_dataset.width * src_dataset.transform.a / transform.a)
        dstrect.attrib['ySize'] = str(src_dataset.height * src_dataset.transform.e / transform.e)

        if src_dataset.nodata is not None:
            nodata_elem = ET.SubElement(simplesource, 'NODATA')
            nodata_elem.text = str(src_dataset.nodata)

    if all(MaskFlags.per_dataset in flags for flags in src_dataset.mask_flag_enums):
        maskband = ET.SubElement(vrtdataset, 'MaskBand')
        vrtrasterband = ET.SubElement(maskband, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = 'Byte'

        simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        sourcefilename = ET.SubElement(simplesource, 'SourceFilename')
        sourcefilename.attrib['relativeToVRT'] = "0"
        sourcefilename.text = vsi_path(parse_path(src_dataset.name))

        sourceband = ET.SubElement(simplesource, 'SourceBand')
        sourceband.text = 'mask,1'
        sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(width)
        sourceproperties.attrib['RasterYSize'] = str(height)
        sourceproperties.attrib['dataType'] = 'Byte'
        sourceproperties.attrib['BlockYSize'] = str(block_shape[0])
        sourceproperties.attrib['BlockXSize'] = str(block_shape[1])
        srcrect = ET.SubElement(simplesource, 'SrcRect')
        srcrect.attrib['xOff'] = '0'
        srcrect.attrib['yOff'] = '0'
        srcrect.attrib['xSize'] = str(src_dataset.width)
        srcrect.attrib['ySize'] = str(src_dataset.height)
        dstrect = ET.SubElement(simplesource, 'DstRect')
        dstrect.attrib['xOff'] = str((src_dataset.transform.xoff - transform.xoff) / transform.a)
        dstrect.attrib['yOff'] = str((src_dataset.transform.yoff - transform.yoff) / transform.e)
        dstrect.attrib['xSize'] = str(src_dataset.width)
        dstrect.attrib['ySize'] = str(src_dataset.height)
    return ET.tostring(vrtdataset)
