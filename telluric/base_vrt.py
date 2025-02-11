"""Base VRT class and fuctions."""

import importlib.resources
import lxml.etree as ET

from xml.dom import minidom
from rasterio.dtypes import _gdal_typename, check_dtype
try:
    from rasterio._path import _parse_path, _vsi_path
except ImportError:
    from rasterio.path import parse_path as _parse_path, vsi_path as _vsi_path


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def load_scheme():
    ref = importlib.resources.files('telluric') / 'gdalvrt.xsd'
    with importlib.resources.as_file(ref) as path:
        with path.open() as f:
            scheme_doc = ET.parse(f)
            return ET.XMLSchema(scheme_doc)


class BaseVRT:
    schema = load_scheme()

    def __init__(self, width=None, height=None, crs=None, affine=None):
        self.vrtdataset = ET.Element('VRTDataset')
        self.set_width(width)
        self.set_height(height)
        self.set_srs(crs)
        self.set_affine(affine)

    def set_width(self, width):
        if width is not None:
            self.vrtdataset.attrib['rasterXSize'] = str(width)

    def set_height(self, height):
        if height is not None:
            self.vrtdataset.attrib['rasterYSize'] = str(height)

    def set_srs(self, crs):
        if crs is not None:
            if not hasattr(self, 'srs_element'):
                self.srs_element = ET.SubElement(self.vrtdataset, 'SRS')
            self.srs_element.text = crs.wkt if crs else ""

    def set_affine(self, affine):
        if affine is not None:
            if not hasattr(self, 'geotransform'):
                self.geotransform = ET.SubElement(self.vrtdataset, 'GeoTransform')
            self.geotransform.text = ','.join([str(v) for v in affine.to_gdal()])

    def add_metadata(self, **kwargs):
        items = kwargs.pop("items")
        metadata_tag = ET.SubElement(self.vrtdataset, 'Metadata')
        for key, val in kwargs.items():
            metadata_tag.attrib[key] = val
        for key, val in items.items():
            mdi = ET.SubElement(metadata_tag, "MDI")
            mdi.attrib["key"] = key
            mdi.text = val

    def add_mask_band(self, dtype):
        maskband = ET.SubElement(self.vrtdataset, 'MaskBand')
        vrtrasterband = ET.SubElement(maskband, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = _gdal_typename(dtype) if check_dtype(dtype) else dtype
        return vrtrasterband

    def add_band(self, dtype, band_idx, color_interp,
                 nodata=None, hidenodata=False):
        vrtrasterband = ET.SubElement(self.vrtdataset, 'VRTRasterBand')
        dtype = dtype if isinstance(dtype, str) else dtype.name
        vrtrasterband.attrib['dataType'] = _gdal_typename(dtype) if check_dtype(dtype) else dtype
        vrtrasterband.attrib['band'] = str(band_idx)

        if nodata is not None:
            nodatavalue = ET.SubElement(vrtrasterband, 'NoDataValue')
            nodatavalue.text = str(nodata)

            if hidenodata:
                hidenodatavalue = ET.SubElement(vrtrasterband, 'HideNoDataValue')
                hidenodatavalue.text = "1" if hidenodata else "0"

        colorinterp = ET.SubElement(vrtrasterband, 'ColorInterp')
        colorinterp.text = color_interp.capitalize()

        return vrtrasterband

    def add_band_complexsource(
        self, vrtrasterband, band_idx, dtype, relative_to_vrt,
        file_name, rasterxsize, rasterysize, blockxsize=None, blockysize=None,
        src_rect=None, dst_rect=None, nodata=None
    ):
        complexsource = ET.SubElement(vrtrasterband, 'ComplexSource')
        self._setup_band_complexsource(
            complexsource, band_idx, dtype, relative_to_vrt, file_name,
            rasterxsize, rasterysize, blockxsize, blockysize, nodata)
        srcrect_element = ET.SubElement(complexsource, 'SrcRect')
        self._setup_rect(srcrect_element, src_rect.col_off, src_rect.row_off,
                         src_rect.width, src_rect.height)
        dstrect_element = ET.SubElement(complexsource, 'DstRect')
        self._setup_rect(dstrect_element, dst_rect.col_off, dst_rect.row_off,
                         dst_rect.width, dst_rect.height)
        usemaskband = ET.SubElement(complexsource, 'UseMaskBand')
        usemaskband.text = 'true'
        return complexsource, srcrect_element, dstrect_element

    def _setup_band_complexsource(
        self, complexsource, band_idx, dtype, relative_to_vrt, file_name,
        rasterxsize, rasterysize, blockxsize, blockysize, nodata
    ):
        sourcefilename = ET.SubElement(complexsource, 'SourceFilename')
        sourcefilename.attrib['relativeToVRT'] = "1" if relative_to_vrt else "0"
        sourcefilename.text = _vsi_path(_parse_path(file_name))
        sourceband = ET.SubElement(complexsource, 'SourceBand')
        sourceband.text = str(band_idx)
        sourceproperties = ET.SubElement(complexsource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(rasterxsize)
        sourceproperties.attrib['RasterYSize'] = str(rasterysize)
        if blockxsize is not None and blockysize is not None:
            sourceproperties.attrib['BlockXSize'] = str(blockxsize)
            sourceproperties.attrib['BlockYSize'] = str(blockysize)
        dtype = dtype if isinstance(dtype, str) else dtype.name
        sourceproperties.attrib['DataType'] = _gdal_typename(dtype) if check_dtype(dtype) else dtype

        # this code was originally in rasterio code but it fails scheme validation
        # so till we figure it out I leave it commented out

        # if nodata is not None:
        #     nodata_elem = ET.SubElement(complexsource, 'NODATA')
        #     nodata_elem.text = str(nodata)

    def _setup_rect(self, sub_element, xoff, yoff, xsize, ysize):
        sub_element.attrib['xOff'] = str(xoff)
        sub_element.attrib['yOff'] = str(yoff)
        sub_element.attrib['xSize'] = str(xsize)
        sub_element.attrib['ySize'] = str(ysize)

    def tostring(self, validate=True):
        if validate:
            self.is_valid()
        return ET.tostring(self.vrtdataset)

    def prettified(self):
        return prettify(self.vrtdataset)

    def validate(self):
        return self.schema.validate(self.vrtdataset)

    def is_valid(self):
        self.schema.assertValid(self.vrtdataset)
