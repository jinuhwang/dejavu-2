# syntax = docker/dockerfile:1.0-experimental
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

USER root

RUN apt update && apt install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux \
    ffmpeg \
    gnuplot \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

RUN git config --global --add safe.directory /workspace

ENV LC_ALL=C.UTF-8
WORKDIR /workspace