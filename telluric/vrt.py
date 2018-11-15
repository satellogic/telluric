"""borrowed from Rasterio"""

import xml.etree.ElementTree as ET

from rasterio.dtypes import _gdal_typename
from rasterio.enums import MaskFlags
from rasterio.path import parse_path, vsi_path


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
