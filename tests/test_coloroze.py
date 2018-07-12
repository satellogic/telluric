import pytest

import numpy as np
import affine
import telluric as tl
from telluric.constants import WGS84_CRS
import matplotlib.pyplot as plt
from common_for_tests import (multi_raster_16b, multi_raster_8b,
                              hyper_raster, hyper_raster_with_no_data,
                              multi_raster_with_no_data)



def test_colorize_jet():
    raster = tl.GeoRaster2(image=np.array(range(256), dtype=np.uint8).reshape((1, 16, 16)),
                           band_names=['red'],
                           crs=WGS84_CRS,
                           affine=affine.Affine(2, 0, 0, 0, 1, 0))

    heatmap = raster.colorize('jet')
    assert(np.array_equal(heatmap.band_names, ['red', 'green', 'blue']))
    assert(np.array_equal(heatmap.image.data[:, 0, 0], [0, 0, 0]))  # nodata remains nodata
    assert(np.array_equal(heatmap.image.mask[:, 0, 0], [True, True, True]))  # nodata remains nodata
    assert(np.array_equal(heatmap.image.data[:, 0, 1], [0, 0, 127]))  # blue
    assert(np.array_equal(heatmap.image.mask[:, 0, 1], [False, False, False]))  # blue
    assert(np.array_equal(heatmap.image.data[:, heatmap.height - 1, heatmap.width - 1],
                                    [127, 0, 0]))  # red
    assert(np.array_equal(heatmap.image.mask[:, heatmap.height - 1, heatmap.width - 1],
                                    [False, False, False]))  # red


def test_colorize_jet_with_range():
    raster = tl.GeoRaster2(image=np.array(range(256), dtype=np.uint8).reshape((1, 16, 16)),
                           band_names=['red'],
                           crs=WGS84_CRS,
                           affine=affine.Affine(2, 0, 0, 0, 1, 0))

    heatmap = raster.colorize('jet', vmin=-1, vmax=1)
    assert(np.array_equal(heatmap.band_names, ['red', 'green', 'blue']))
    assert(np.array_equal(heatmap.image.data[:, 0, 0], [0, 0, 0]))  # nodata remains nodata
    assert(np.array_equal(heatmap.image.mask[:, 0, 0], [True, True, True]))  # nodata remains nodata
    assert((heatmap.image.data[0, heatmap.image.data[0] > 0] == 127).all())
    assert((heatmap.image.data[1, heatmap.image.data[1] > 0] == 0).all())
    assert((heatmap.image.data[2, heatmap.image.data[2] > 0] == 0).all())
    mask = heatmap.image.mask
    assert(len(mask[mask == np.True_]) == 3)


@pytest.mark.parametrize("raster", [multi_raster_16b(),
                                    multi_raster_8b(),
                                    hyper_raster(),
                                    hyper_raster_with_no_data(),
                                    multi_raster_with_no_data()])
@pytest.mark.parametrize("colormap", plt.cm.datad.keys())
def test_colorize_works_for_all(raster, colormap):
    with pytest.warns(tl.georaster.GeoRaster2Warning, match='Using the first band to colorize the raster'):
        heatmap = raster.colorize('jet')
    assert(np.array_equal(heatmap.band_names, ['red', 'green', 'blue']))

