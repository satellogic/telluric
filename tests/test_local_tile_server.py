
from os import path
from common_for_tests import make_test_raster
from tornado.testing import gen_test, AsyncHTTPTestCase

import telluric as tl
from telluric.util.local_tile_server import TileServer, make_app

# tiles = [(131072, 131072, 18)]
tiles = [(131072, 131072, 18), (65536, 65536, 17), (32768, 32768, 16), (16384, 16384, 15)]


class TestFCLocalTileServer(AsyncHTTPTestCase):

    def get_app(self):
        rasters = [
            make_test_raster(i, band_names=["band%i" % i], height=300, width=400)
            for i in range(3)
        ]

        self.fc = tl.FeatureCollection([tl.features.GeoFeatureWithRaster(r, {}) for r in rasters])
        TileServer.add_object(self.fc, self.fc.envelope)
        return make_app(TileServer.objects)

    def test_server_is_alive(self):
        response = self.fetch('/ok')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"i'm alive")

    def test_raster_collection_merges_data(self):
        for tile in tiles:
            uri = "/%i/%i/%i/%i.png" % (id(self.fc), *tile)
            response = self.fetch(uri)
            self.assertEqual(response.code, 200)
            self.assertNotEqual(response.body, b"")


class TestRasterLocalTileServer(AsyncHTTPTestCase):

    def get_app(self):
        self.raster = make_test_raster(1, band_names=["band1"], height=300, width=400)
        TileServer.add_object(self.raster, self.raster.footprint())
        return make_app(TileServer.objects)

    def test_server_is_alive(self):
        response = self.fetch('/ok')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"i'm alive")

    def test_raster_collection_merges_data(self):
        for tile in tiles:
            uri = "/%i/%i/%i/%i.png" % (id(self.raster), *tile)
            response = self.fetch(uri)
            self.assertEqual(response.code, 200)
            self.assertNotEqual(response.body, b"")
            raster = tl.GeoRaster2.from_bytes(response.body, self.raster.affine, self.raster.crs)
            self.assertEqual(raster.shape, (3, 256, 256))
