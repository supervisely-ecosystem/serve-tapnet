FROM supervisely/base-py-sdk:6.73.418

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y ffmpeg
RUN apt-get install -y libopenexr-dev

RUN mkdir -p checkpoint
RUN wget https://storage.googleapis.com/dm-tapnet/checkpoint.npy -P checkpoint/

RUN git clone https://github.com/deepmind/tapnet.git
RUN export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
WORKDIR tapnet
RUN git clone https://github.com/google-research/kubric.git
RUN pip3 install jax==0.4.7
WORKDIR /

COPY tapnet/evaluation_datasets.py /tapnet/

RUN pip3 install --no-cache-dir absl-py~=1.2.0 \
    ml-collections~=0.1.1 \
    matplot~=0.1.9 \
    matplotlib~=3.4.3 \
    matplotlib-inline~=0.1.2 \
    numpy~=1.22.4 \
    pyquaternion~=0.9.9 \
    munch~=2.5.0 \
    typing~=3.7.4.3 \
    dataclasses~=0.6 \
    etils~=0.6.0 \
    tensorflow~=2.8.2 \
    trimesh~=3.12.9 \
    tqdm~=4.64.0 \
    requests~=2.28.1 \
    singledispatchmethod~=1.0 \
    openexr~=1.3.8 \
    scikit-learn~=1.1.1 \
    imageio~=2.19.5 \
    setuptools~=61.2.0 \
    mediapy~=1.0.3 \
    scipy~=1.8.1 \
    optax~=0.1.3 \
    chex~=0.1.3 \
    jaxline~=0.0.5 \
    pillow~=9.2.0 \
    dm-haiku~=0.0.7 \
    tensorflow_datasets~=4.6.0 \
    tensorflow_graphics~=2021.12.3 \
    typing_extensions~=4.3.0 \
    einshape @ git+https://github.com/deepmind/einshape \
    jax==0.4.7 \
    jaxlib==0.4.7


RUN pip3 install supervisely==6.73.486

LABEL python_sdk_version="6.73.486"