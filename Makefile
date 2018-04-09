DOCKER_IMAGE?=telluric:latest

.PHONY: build dockershell test test-fast test-manual devshell

build:
	docker build --pull -t ${DOCKER_IMAGE} .

dockershell:
	docker run --rm -v `pwd`:/usr/src -it ${DOCKER_IMAGE} /bin/bash

test:
	docker run --rm ${DOCKER_IMAGE} python3 -m pytest

test-fast:
	docker run --rm -v `pwd`:/usr/src ${DOCKER_IMAGE} python3 -m pytest --color=yes

test-manual:
	docker run --rm --env TEST_MANUAL=1 -v `pwd`:/usr/src ${DOCKER_IMAGE} python3 -m pytest -k ManualTest -s
