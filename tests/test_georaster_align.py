import pytest
from telluric.georaster import mercator_zoom_to_resolution, GeoRaster2
from common_for_tests import make_test_raster
import mercantile
from telluric.constants import WGS84_CRS
import numpy as np


def test_mercator_zoom_to_resolution():
    assert mercator_zoom_to_resolution == {0: 156543.03392804097,
                                            1: 78271.51696402048,
                                            2: 39135.75848201024,
                                            3: 19567.87924100512,
                                            4: 9783.93962050256,
                                            5: 4891.96981025128,
                                            6: 2445.98490512564,
                                            7: 1222.99245256282,
                                            8: 611.49622628141,
                                            9: 305.748113140705,
                                            10: 152.8740565703525,
                                            11: 76.43702828517625,
                                            12: 38.21851414258813,
                                            13: 19.109257071294063,
                                            14: 9.554628535647032,
                                            15: 4.777314267823516,
                                            16: 2.388657133911758,
                                            17: 1.194328566955879,
                                            18: 0.5971642834779395}

def test_mercator_upper_zoom_level():
    raster = make_test_raster()
    assert raster.mercator_upper_zoom_level() == 18

def test_get_mercator_bouding_box():
    raster = make_test_raster()
    alinged_bounding_box = raster.mercator_alligned_bouding_box()
    assert raster.footprint().within(alinged_bounding_box)
    validate_mercator_bounding_box(alinged_bounding_box, raster.mercator_upper_zoom_level())

def validate_mercator_bounding_box(vector, zoom_level):
     #verfying the ul point is also the ul point of a tile
    bounds = vector.get_shape(WGS84_CRS).bounds
    ul_tile = mercantile.tile( bounds[0], bounds[3], zoom=zoom_level)
    ul_of_tile = mercantile.ul(ul_tile)
    assert ul_of_tile.lat - bounds[3] < 0.00001
    assert ul_of_tile.lng - bounds[0] < 0.00001


def test_align_to_mercator():
    raster = make_test_raster(width=200, height=200, value=300, band_names=["red"], dtype=np.uint8)
    expected_resultion  = mercator_zoom_to_resolution[raster.mercator_upper_zoom_level()]
    alinged_raster = raster.align_raster_to_mercator_tiles()
    assert expected_resultion == alinged_raster.resolution()
    assert alinged_raster.resolution() == mercator_zoom_to_resolution[alinged_raster.mercator_upper_zoom_level()]
    # assert raster.mercator_alligned_bouding_box().almost_equals(alinged_raster.mercator_alligned_bouding_box())
    assert alinged_raster.footprint().almost_equals(alinged_raster.mercator_alligned_bouding_box())



