# Dockerfile to run Cloud Optimize Geotiff validation which requires GDAL2.2 and above
# example of use: 
#   build: docker build . -t gdal2.2
#   run:`docker run  -v `pwd`:/usr/local/src -it gdal2.2 path/to?geotiff.tif`
FROM geographica/gdal2
RUN  apt-get update
RUN  apt-get -y install python3-pip
RUN  pip3 install --upgrade pip
RUN  pip3 install ipython
RUN  pip3 install gdal
COPY validate_cloud_optimized_geotiff.py .
ENTRYPOINT [ "python3", "../validate_cloud_optimized_geotiff.py" ]
WORKDIR /usr/local/src