import asyncio
import folium
import rasterio
import tornado.web
import concurrent.futures

from tornado import gen
from threading import Thread, Lock
from tornado.ioloop import IOLoop
from rasterio.enums import Resampling
from telluric.constants import WGS84_CRS
from tornado.httpserver import HTTPServer
from tornado.concurrent import run_on_executor


executor = concurrent.futures.ThreadPoolExecutor(50)


class TileServerHandler(tornado.web.RequestHandler):
    _tread_pool = executor

    def initialize(self, rasters, resampling):
        self.rasters = rasters
        self.resampling = resampling

    @gen.coroutine
    def get(self, raster_id, x, y, z):
        png_tile = yield self._get_png_tile(int(raster_id), int(x), int(y), int(z))
        if png_tile:
            self.set_header("Content-type", "image/png")
            self.finish(png_tile)
        else:
            self.send_error(404)

    @run_on_executor(executor='_tread_pool')
    def _get_png_tile(self, raster_id, x, y, z):
        raster = self.rasters[raster_id]
        with rasterio.Env():
            return raster.get_tile(x, y, z, resampling=self.resampling).to_png()


class OKHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("i'm alive")


def make_app(rasters, resampling):
    uri = r'/(\d+)/(\d+)/(\d+)/(\d+)\.png'
    return tornado.web.Application([
        (uri, TileServerHandler, dict(rasters=rasters, resampling=resampling)),
        (r'/ok', OKHandler),
    ])


def _run_app(rasters, resampling, port=4000):
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = HTTPServer(make_app(rasters, resampling))
    app.listen(port, '0.0.0.0')
    IOLoop.current().start()


rasters_lock = Lock()


class TileServer:
    rasters = {}  # type:dict
    running_app = None

    @classmethod
    def folium_client(cls, raster, base_map="Stamen Terrain", port=4000):
        shape = raster.footprint().get_shape(WGS84_CRS)
        mp = folium.Map(tiles=base_map)

        folium.raster_layers.TileLayer(
            tiles=cls.server_url(raster, port),
            attr="raster: %s" % raster._filename,
            overlay=True
        ).add_to(mp)

        mp.fit_bounds([shape.bounds[:1:-1], shape.bounds[1::-1]])

        return mp

    @classmethod
    def server_url(cls, raster, port):
        return "http://localhost:%s/%s/{x}/{y}/{z}.png" % (port, id(raster))

    @classmethod
    def run_tileserver(cls, raster, resampling=Resampling.cubic, port=4000):
        cls.add_raster(raster)
        if cls.running_app is None:
            try:
                cls.running_app = Thread(None, _run_app, args=(cls.rasters, resampling, port),
                                         name='TileServer', daemon=True)
                cls.running_app.start()
                return cls.running_app
            except Exception as e:
                print(e)
                cls.running_app = None
                raise e

    @classmethod
    def add_raster(cls, raster):
        with rasters_lock:
            cls.rasters[id(raster)] = raster
