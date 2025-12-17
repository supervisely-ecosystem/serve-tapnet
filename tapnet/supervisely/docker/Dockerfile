FROM supervisely/base-py-sdk:6.72.8

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
RUN pip3 install -r requirements.txt
RUN pip3 install jax==0.4.7
WORKDIR /

RUN pip3 install supervisely==6.73.486

COPY evaluation_datasets.py /tapnet/

LABEL python_sdk_version="6.73.486"
