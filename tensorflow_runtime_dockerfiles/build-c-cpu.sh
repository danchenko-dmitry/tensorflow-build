#!/bin/bash

docker build --target=cling --build-arg TENSORFLOW_PACKAGE=tensorflow -t tensorflow --platform linux/amd64 --progress plain -f cpu-c.Dockerfile . 
