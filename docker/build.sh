#!/bin/bash

cp ../requirements.txt .
# For debugging, use --progress=plain option
DOCKER_BUILDKIT=1 docker build . -t dejavu:base
rm requirements.txt
