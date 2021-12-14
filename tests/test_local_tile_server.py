import pytest
from unittest import mock

pytest.importorskip("tornado")  # noqa: E402
pytest.importorskip("matplotlib")  # noqa: E402

from common_for_tests import make_test_raster
from tornado.testing import AsyncHTTPTestCase
from tornado.concurrent import Future

import telluric as tl
from telluric.util.local_tile_server import TileServer, make_app, TileServerHandler

tiles = [(131072, 131072, 18), (65536, 65536, 17), (32768, 32768, 16), (16384, 16384, 15)]

rasters = [
    make_test_raster(i, band_names=["band%i" % i], height=300, width=400)
    for i in range(3)
]


class TestFCLocalTileServer(AsyncHTTPTestCase):

    def get_app(self):

        self.fc = tl.FeatureCollection([tl.GeoFeature.from_raster(r, {}) for r in rasters])
        TileServer.add_object(self.fc, self.fc.envelope)
        return make_app(TileServer.objects)

    def test_server_is_alive(self):
        response = self.fetch('/ok')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"i'm alive")

    @mock.patch.object(TileServerHandler, '_get_raster_png_tile')
    @mock.patch.object(TileServerHandler, '_merge_rasters')
    def test_raster_collection_merges_data(self, mock_merge, mock_get_tile):
        future_1 = Future()
        future_1.set_result(rasters[1])
        mock_merge.return_value = future_1
        future_2 = Future()
        future_2.set_result(rasters[2])
        mock_get_tile.return_value = future_2
        for tile in tiles:
            uri = "/%i/%i/%i/%i.png" % (id(self.fc), tile[0], tile[1], tile[2])
            response = self.fetch(uri)
            self.assertEqual(response.code, 200)
            self.assertNotEqual(response.body, b"")
            self.assertEqual(mock_get_tile.call_count, 3)
            self.assertEqual(mock_merge.call_count, 1)
            self.assertEqual(mock_merge.call_args[0][1], tile[2])
            for r in mock_merge.call_args[0][0]:
                self.assertIsInstance(r, tl.GeoRaster2)
            self.assertEqual(len(mock_merge.call_args[0][0]), 3)
            mock_get_tile.reset_mock()
            mock_merge.reset_mock()


class TestRasterLocalTileServer(AsyncHTTPTestCase):

    def get_app(self):
        self.raster = rasters[1]
        TileServer.add_object(self.raster, self.raster.footprint())
        return make_app(TileServer.objects)

    def test_server_is_alive(self):
        response = self.fetch('/ok')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"i'm alive")

    def test_raster_collection_merges_data(self):
        for tile in tiles:
            uri = "/%i/%i/%i/%i.png" % (id(self.raster), tile[0], tile[1], tile[2])
            response = self.fetch(uri)
            self.assertEqual(response.code, 200)
            self.assertNotEqual(response.body, b"")
            raster = tl.GeoRaster2.from_bytes(response.body, self.raster.affine, self.raster.crs)
            self.assertEqual(raster.shape, (3, 256, 256))
