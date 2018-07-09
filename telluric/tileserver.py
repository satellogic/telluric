from tornado.ioloop import IOLoop
from tornado.web import url, Application, RequestHandler
from tornado import gen
from tornado.httpserver import HTTPServer





class MainHandler(RequestHandler):
    def get(self):
        self.write("ok")


class TilesHandler(RequestHandler):
    def initialize(self, fc):
        self.fc = fc

    @gen.coroutine
    def _get_tile(self, z, x, y):
        return self.fc.get_tile(x,y,z, masked=False)

    @gen.coroutine
    def get(self, z, x, y):
        x, y, z = int(x), int(y), int(z)
        tile = yield self._get_tile(z, x, y)
        if tile:
            self.write(tile.to_png())
            self.set_header("Content-type", "image/png")
        else:
            self.send_error(404)


class TileServer():

    def __init__(self, feature_collections, host_name='localhost', port=4444):
        self.feature_collections = feature_collections

        self.port = port
        self.host_name = host_name
        self.server = HTTPServer(self._make_app())

    def run(self):
        """Run server only in multiprocess."""
        print("starting server on port %s" % self.port)
        self.server.bind(self.port)
        self.server.start(0)
        IOLoop.current().start()

    def start(self):
        """Start server only on a Jupyter Notebook."""
        self.server.listen(self.port)

    def stop(self):
        """Stop server only on a Jupyter Notebook."""
        self.server.stop()

    def _make_app(self):
        return Application([
            url(r"/", MainHandler),
            url(r"/tiles/([0-9]{1,3})/([0-9]{1,10})/([0-9]{1,10})", TilesHandler,
                dict(fc=self.feature_collections), name='tiles'),
        ])

    def get_start_point(self):
        first_fc = next(iter(self.feature_collections[0]))
        start_point = first_fc.get_shape(WGS84_CRS).centroid
        return start_point.y, start_point.x

    def get_url(self):
        return "http://{host}:{port}/tiles/{{z}}/{{x}}/{{y}}".format_map({'host': self.host_name,
                                                                          'port': self.port})

    def get_folium_client(self, **folium_kwargs):
        import folium
        tiles = self.get_url()
        layer = folium.TileLayer(tiles=tiles, attr='telluric')
        folium_args = {
            'location': self.get_start_point(),
            'zoom_start': 8,
        }
        folium_args.update(folium_kwargs)
        m = folium.Map(**folium_args)
        layer.add_to(m)
        return m
