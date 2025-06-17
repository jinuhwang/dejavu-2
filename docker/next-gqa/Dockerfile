FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt update && apt install -y \
    build-essential \
    git

ADD requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

RUN python -m pip install \
    torch==1.8.1 \
    hydra-core \
    lightning

WORKDIR /workspace
