#!/bin/bash

docker build --target=cling --build-arg TENSORFLOW_PACKAGE=tensorflow -t tensorflow-gpu --platform linux/amd64 --progress plain -f gpu-c.Dockerfile . 
