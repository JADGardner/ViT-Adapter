FROM docker.io/nvidia/cuda:11.6.2-devel-ubuntu20.04

USER root

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    python3.8-dev \
    python3-pip \
    wget \ 
    curl \
    unzip \

# Create non root user and setup environment.
RUN useradd -m -d /home/user -u 1000 user

# make .ssh directory and set permissions for user
RUN mkdir -p /home/user/.ssh && \
    chown -R user:user /home/user/.ssh && \
    chmod 700 /home/user/.ssh

RUN chown -R user:user /usr/local/

# Switch to new uer and workdir.
USER 1000:1000
WORKDIR /home/user

RUN pip install -U pylint
RUN pip install -U black
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install scipy
RUN pip install tqdm
RUN pip install timm==0.4.12
RUN pip install mmdet==2.22.0 # for Mask2Former
RUN pip install mmsegmentation==0.20.2