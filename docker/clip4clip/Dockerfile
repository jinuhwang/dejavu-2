# Use PyTorch 1.7.1
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

USER root
WORKDIR /root

# Resolve GPG:NO_PUBKEY error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install ftfy regex tqdm \
    && pip install numpy==1.20.3 \
    && pip install opencv-python boto3 requests pandas

RUN pip install hydra omegaconf

WORKDIR /workspace