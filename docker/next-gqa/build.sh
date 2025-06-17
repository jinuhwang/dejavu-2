#!/bin/bash

set -e

cp ../../third_parties/NExT-GQA/requirements.txt .
# For debugging, use --progress=plain option
DOCKER_BUILDKIT=1 docker build . -t next-gqa:latest
rm requirements.txt
