"""Base VRT class and fuctions."""

import lxml.etree as ET

from collections import namedtuple
from xml.dom import minidom
from rasterio.enums import MaskFlags
from rasterio.dtypes import _gdal_typename, check_dtype
from rasterio.path import parse_path, vsi_path


RectElement = namedtuple('RectElelment', 'xoff yoff xsize ysize')


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def add_sub_element(element, name, text=None, **attributes):
    sub_element = ET.SubElement(element, name)
    for attr, val in attributes.items():
        sub_element.attrib[attr] = val
    if text is not None:
        sub_element.text = text
    return sub_element


class BaseVRT:
    def __init__(self, width, height, srs, affine,
                 nodata=None, background=None, hidenodata=False):
        self.nodata = nodata
        self.background = background
        self.hidenodata = hidenodata
        self.height = height
        self.width = width
        self.affine = affine
        self.vrtdataset = ET.Element('VRTDataset')
        self.vrtdataset.attrib['rasterXSize'] = str(width)
        self.vrtdataset.attrib['rasterYSize'] = str(height)
        srs_element = ET.SubElement(self.vrtdataset, 'SRS')
        srs_element.text = srs
        geotransform = ET.SubElement(self.vrtdataset, 'GeoTransform')
        geotransform.text = ','.join([str(v) for v in affine.to_gdal()])
        self.root = self.vrtdataset
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = ET.SubElement(self.root, 'Metadata')
        return self._metadata

    def add_metadata_attributes(self, **attributes):
        for attr, val in attributes.items():
            self.metadata.attrib[attr] = val

    def add_entity_to_metadata(self, name, text=None, **attributes):
        sub_element = ET.SubElement(self.metadata, name)
        for attr, val in attributes.items():
            sub_element.attrib[attr] = val
        if text is not None:
            sub_element.text = text
        return sub_element

    def add_mask_band(self):
        maskband = ET.SubElement(self.vrtdataset, 'MaskBand')
        vrtrasterband = ET.SubElement(maskband, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = 'Byte'
        return vrtrasterband

        # simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        # self._setup_band_simplesource(simplesource, 'mask,1', 'Byte', False, self.src_dataset.name,
        #                               self.width, self.height, block_shape[1], block_shape[0])
        # srcrect = ET.SubElement(simplesource, 'SrcRect')
        # self._setup_rect(srcrect, 0, 0, self.src_dataset.width, self.src_dataset.height)
        # dstrect = ET.SubElement(simplesource, 'DstRect')
        # xoff = (self.src_dataset.transform.xoff - self.affine.xoff) / self.affine.a
        # yoff = (self.src_dataset.transform.yoff - self.affine.yoff) / self.affine.e
        # xsize = self.src_dataset.width
        # ysize = self.src_dataset.height
        # self._setup_rect(dstrect, xoff, yoff, xsize, ysize)

    def add_band(self, dtype, band_name, color_interp,
                 nodata=None, nodata_val=None, hidenodata=False):
        vrtrasterband = ET.SubElement(self.vrtdataset, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = _gdal_typename(dtype) if check_dtype(dtype) else dtype
        vrtrasterband.attrib['band'] = str(band_name)

        if nodata_val is not None:
            nodatavalue = ET.SubElement(vrtrasterband, 'NoDataValue')
            nodatavalue.text = str(self.nodata_val)

            if hidenodata:
                hidenodatavalue = ET.SubElement(vrtrasterband, 'HideNoDataValue')
                hidenodatavalue.text = "1"

        colorinterp = ET.SubElement(vrtrasterband, 'ColorInterp')
        colorinterp.text = color_interp.capitalize()

        # if background is not None:
        #     simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        #     self._setup_band_simplesource(simplesource, band_name, dtype, False, background.name,
        #                                   self.width, self.height, block_shape[1], block_shape[0])
        #     srcrect = ET.SubElement(simplesource, 'SrcRect')
        #     self._setup_rect(srcrect, 0, 0, background.width, background.height)
        #     dstrect = ET.SubElement(simplesource, 'DstRect')
        #     self._setup_rect(dstrect, 0, 0, self.width, self.height)

        return vrtrasterband

    def add_band_simplesource(self, vrtrasterband, band_name, dtype, relative_to_vrt,
                              file_name, rasterxsize, rasterysize, blockxsize, blockysize,
                              src_rect, dst_rect, nodata=None
                              ):
        simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        self._setup_band_simplesource(simplesource, band_name, dtype, relative_to_vrt, file_name,
                                      rasterxsize, rasterysize, blockxsize, blockysize, nodata)
        srcrect_element = ET.SubElement(simplesource, 'SrcRect')
        self._setup_rect(srcrect_element, src_rect.xoff, src_rect.yoff,
                         src_rect.xsize, src_rect.ysize)
        dstrect_element = ET.SubElement(simplesource, 'DstRect')
        # xoff = (self.src_dataset.transform.xoff - self.affine.xoff) / self.affine.a
        # yoff = (self.src_dataset.transform.yoff - self.affine.yoff) / self.affine.e
        # xsize = self.src_dataset.width * self.src_dataset.transform.a / self.affine.a
        # ysize = self.src_dataset.height * self.src_dataset.transform.e / self.affine.e
        self._setup_rect(dstrect_element, dst_rect.xoff, dst_rect.yoff,
                         dst_rect.xsize, dst_rect.ysize)
        return simplesource, srcrect_element, dstrect_element

    def _setup_band_simplesource(self, simplesource, band_name, dtype, relative_to_vrt, file_name,
                                 rasterxsize, rasterysize, blockxsize, blockysize, nodata):
        sourcefilename = ET.SubElement(simplesource, 'SourceFilename')
        sourcefilename.attrib['RelativeToVRT'] = "1" if relative_to_vrt else "0"
        sourcefilename.text = vsi_path(parse_path(file_name))
        sourceband = ET.SubElement(simplesource, 'SourceBand')
        sourceband.text = str(band_name)
        sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(rasterxsize)
        sourceproperties.attrib['RasterYSize'] = str(rasterysize)
        sourceproperties.attrib['DataType'] = _gdal_typename(dtype) if check_dtype(dtype) else dtype
        sourceproperties.attrib['BlockXSize'] = str(blockxsize)
        sourceproperties.attrib['BlockYSize'] = str(blockysize)
        if nodata is not None:
            nodata_elem = ET.SubElement(simplesource, 'NODATA')
            nodata_elem.text = str(nodata)



    def _setup_rect(self, sub_element, xoff, yoff, xsize, ysize):
        sub_element.attrib['xOff'] = str(xoff)
        sub_element.attrib['yOff'] = str(yoff)
        sub_element.attrib['xSize'] = str(xsize)
        sub_element.attrib['ySize'] = str(ysize)

    def tostring(self):
        return ET.tostring(self.vrtdataset)

    def prettified(self):
        return prettify(self.vrtdataset)

    def validate(self):
        pass

    def is_valid(self):
        pass