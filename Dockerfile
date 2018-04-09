# Development environment for telluric
# Analyzed with https://www.fromlatest.io/
FROM debian:jessie-backports
LABEL maintainer="Juan Luis Cano <juanlu@satellogic.com>"

RUN set -x \
    && apt-get update \
    && apt-get install --no-install-recommends -t jessie-backports -y \
    wget python3-all-dev python3-all python3-tk  \
    ca-certificates wget build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN wget -O - https://bootstrap.pypa.io/get-pip.py | python3
RUN set -x \
   && pip3 install pip -U \
   && pip3 install numpy

# Patch for GDAL vsicull to work of https
RUN mkdir -p /etc/pki/tls/certs/
RUN cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

WORKDIR /usr/src
COPY . /usr/src

RUN pip3 install --editable .[dev]
