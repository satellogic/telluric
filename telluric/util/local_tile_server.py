import asyncio
import os
import concurrent.futures
from threading import Thread, Lock
from collections import namedtuple
import tornado.web
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.concurrent import run_on_executor

try:
    import folium
except ImportError:
    import warnings
    warnings.warn(
        "Visualization dependencies not available, folium client will not work",
        ImportWarning,
        stacklevel=2,
    )

import rasterio
from rasterio.enums import Resampling

import telluric as tl
from telluric.constants import WGS84_CRS
from telluric.constants import MERCATOR_RESOLUTION_MAPPING, WEB_MERCATOR_CRS


raster_executor = concurrent.futures.ThreadPoolExecutor(50)

ServedObj = namedtuple('ServedObj', 'obj footprint')


class TileServerHandler(tornado.web.RequestHandler):
    _thread_pool = raster_executor

    def initialize(self, objects, resampling):
        self.objects = objects
        self.resampling = resampling

    @gen.coroutine
    def get(self, object_id, x, y, z):
        # the import is here to eliminate recursive import
        from telluric.collections import BaseCollection
        object_id, x, y, z = int(object_id), int(x), int(y), int(z)
        obj = self.objects[object_id]
        tile_vector = tl.GeoVector.from_xyz(x, y, z)

        if tile_vector.intersects(obj.footprint):
            if isinstance(obj.obj, tl.GeoRaster2):
                tile = yield self._get_raster_png_tile(obj.obj, x, y, z)
            if isinstance(obj.obj, tl.GeoFeature) and obj.obj.has_raster:
                tile = yield self._get_raster_png_tile(obj.obj.raster(), x, y, z)
            elif isinstance(obj.obj, BaseCollection):
                tile = yield self._get_collection_png_tile(obj.obj, x, y, z)

            if tile:
                self.set_header("Content-type", "image/png")
                self.finish(tile.to_png())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    @run_on_executor(executor='_thread_pool')
    def _get_raster_png_tile(self, raster, x, y, z):
        with rasterio.Env():
            return raster.get_tile(x, y, z, resampling=self.resampling)

    @gen.coroutine
    def _get_collection_png_tile(self, fc, x, y, z):
        rasters = yield gen.multi([self._get_raster_png_tile(f.raster(), x, y, z) for f in fc])  # type: ignore
        if len(rasters) < 1:
            return None
        tile = yield self._merge_rasters(rasters, z)
        return tile

    @run_on_executor(executor='_thread_pool')
    def _merge_rasters(self, rasters, z):
        # the import is here to eliminate recursive import
        from telluric.georaster import merge_all
        actual_roi = rasters[0].footprint()
        merge_params = {
            'dest_resolution': MERCATOR_RESOLUTION_MAPPING[z],
            'ul_corner': (actual_roi.left, actual_roi.top),
            'shape': (256, 256),
            'crs': WEB_MERCATOR_CRS,
        }
        return merge_all(rasters, **merge_params)


class OKHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        answer = yield self.answer()
        self.write(answer)

    @gen.coroutine
    def answer(self):
        return "i'm alive"


rasters_lock = Lock()


class TileServer:
    objects = {}  # type:dict
    running_app = None

    @staticmethod
    def default_port():
        return int(os.getenv("TELLURIC_TILESERVER_PORT", "4000"))

    @classmethod
    def folium_client(cls, obj, bounds, mp=None, capture=None,
                      base_map_name="Stamen Terrain", port=None):
        port = port or cls.default_port()
        shape = bounds.get_shape(WGS84_CRS)
        mp = mp or folium.Map(tiles=base_map_name)

        folium.raster_layers.TileLayer(
            tiles=cls.server_url(obj, port),
            attr=capture,
            overlay=True
        ).add_to(mp)

        mp.fit_bounds([shape.bounds[:1:-1], shape.bounds[1::-1]])

        return mp

    @classmethod
    def server_url(cls, obj, port):
        return "http://localhost:%s/%s/{x}/{y}/{z}.png" % (port, id(obj))

    @classmethod
    def run_tileserver(cls, obj, footprint, resampling=Resampling.nearest, port=None):
        port = port or cls.default_port()
        cls.add_object(obj, footprint)
        if cls.running_app is None:
            try:
                cls.running_app = Thread(None, _run_app, args=(cls.objects, resampling, port),
                                         name='TileServer', daemon=True)
                cls.running_app.start()
                return cls.running_app
            except Exception as e:
                print(e)
                cls.running_app = None
                raise e

    @classmethod
    def add_object(cls, obj, footprint):
        with rasters_lock:
            cls.objects[id(obj)] = ServedObj(obj, footprint)


def make_app(objects, resampling=Resampling.nearest):
    uri = r'/(\d+)/(\d+)/(\d+)/(\d+)\.png'
    return tornado.web.Application([
        (uri, TileServerHandler, dict(objects=objects, resampling=resampling)),
        (r'/ok', OKHandler),
    ])


def _run_app(objects, resampling, port=TileServer.default_port()):
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = HTTPServer(make_app(objects, resampling))
    app.listen(port, '0.0.0.0')
    IOLoop.current().start()
