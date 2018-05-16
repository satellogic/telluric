import os
from tile_server import TileServer
from geostore import azure
import dsnparse
import telluric as tl
from furl import furl
import arrow

cog_url = os.environ.get("RASTER_URL")

def prepare_url(url):
    if url.startswith('azblob://'):
        # If the filename is of the form az://container-name/path/to/raster.tif,
        # generate a temporary access token and append it to the URL
        dsn = dsnparse.parse(url)
        url = azure.generate_url(dsn.host, dsn.path.lstrip("/"), expire=6000)
        urlf = furl(url)
        urlf.args["tt"] = arrow.now().isoformat()
        urlf.args["timeout"] = 30
        url = urlf.url
        print(url)
        return url


url = prepare_url(cog_url)
raster = tl.GeoRaster2.open(url)
footprint = raster.footprint()
feature = tl.GeoFeature(footprint, {'raster_url': url})
feature.crs

features = []
for i in range(50):
    url = prepare_url(cog_url)
    features.append(tl.GeoFeature(footprint, {'raster_url': url}))
fc = tl.FeatureCollection(features)



ts = TileServer(fc)
ts.run()




# feature_collections = os.environ.get("TELLURIC_FEATURE_COLLECTIOS", "./miniserver_demo/ds2_remote.json")
# feature_collections = os.environ.get("TELLURIC_FEATURE_COLLECTIOS", "./ds2.json")
# feature_collections = feature_collections.split(',')

# ts = TileServer(feature_collections)

# print(ts.get_start_point())
# print(ts.get_folium_client())
# ts.run()
