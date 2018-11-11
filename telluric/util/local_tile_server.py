import asyncio
import folium
import rasterio
import tornado.web
from tornado import gen
from threading import Thread, Lock
from tornado.ioloop import IOLoop
from telluric.constants import WGS84_CRS


class TileServerHandler(tornado.web.RequestHandler):
    def initialize(self, rasters):
        self.rasters = rasters

    @tornado.web.asynchronous
    @gen.coroutine
    def get(self, raster_id, x, y, z):
        with rasterio.Env():
            tile = self.rasters[int(raster_id)].get_tile(int(x), int(y), int(z))
        self.write(tile.to_png())
        self.set_header("Content-type", "image/png")
        self.finish()


class OKHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("i'm alive")


def make_app(rasters):
    uri = r'/(\d+)/(\d+)/(\d+)/(\d+)\.png'
    return tornado.web.Application([
        (uri, TileServerHandler, dict(rasters=rasters)),
        (r'/ok', OKHandler),
    ])


def _run_app(rasters, port=4000):
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = make_app(rasters)
    app.listen(port, '0.0.0.0')
    IOLoop.current().start()


rasters_lock = Lock()


class TileServer:
    rasters = {}
    running_app = None

    @classmethod
    def folium_client(cls, raster, base_map="Stamen Terrain", port=4000, zoom_start=13):
        raster_center = raster.center().get_shape(WGS84_CRS)
        # print([raster_center.x, raster_center.y])
        mp = folium.Map(
            tiles=base_map,
            location=[raster_center.x, raster_center.y],
            zoom_start=zoom_start
        )
        # print("http://localhost:%s/%s/{x}/{y}/{z}.png" % (port, id(raster)))
        folium.raster_layers.TileLayer(
            tiles="http://localhost:%s/%s/{x}/{y}/{z}.png" % (port, id(raster)),
            attr="raster %s" % raster._filename
        ).add_to(mp)

        return mp

    @classmethod
    def run_tileserver(cls, raster, port=4000):
        cls.add_raster(raster)
        if cls.running_app is None:
            try:
                cls.running_app = Thread(None, _run_app, args=(cls.rasters, port),
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
