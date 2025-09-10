#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=dejavu
TAG=base
NAME=jwhwang_dejavu
MNT_DIR=/mnt

# Temporary change while repo migration
docker run -it \
    --net=host \
    -v ${SCRIPT_DIR}/..:/workspace \
    -v ${MNT_DIR}/nfs/new:/mnt/raid \
    --gpus all \
    --ulimit core=-1 --privileged \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}:${TAG} bash
