import pycurl
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import time


url = "https://telluriccatalogtest.blob.core.windows.net/test/4Mfile?sp=rl&st=2018-04-16T08%3A30%3A00Z&se=2018-06-17T08%3A30%3A00Z&sv=2017-04-17&sig=NCQM9G8PIeGQEGtdO1fipqZgKjZSI5zVZciIkRFbJUA%3D&sr=b&timeout=10"


def get_with_curl(_id):

    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    # c.setopt(c.RANGE, "95354880-95649791")
    c.setopt(c.RANGE, "1000-17000")
    c.setopt(c.WRITEDATA, buffer)
    c.perform()

    c.close()

    body = buffer.getvalue()

    print(_id, len(body))

def get_with_requests(_id):
    res = requests.get(url, headers={"Range": "bytes=1000-17000"})
    print(_id, len(res.content))
    # time.sleep(1)

args = range(100)
with ThreadPoolExecutor(max_workers=100) as executer:
            tiled_features = list(executer.map(get_with_curl,
                                               args,
                                               timeout=100))
            # tiled_features = list(executer.map(get_with_requests,
                                               # args,
                                               # timeout=100))

