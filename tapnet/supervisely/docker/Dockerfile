FROM supervisely/base-py-sdk:6.72.3

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
WORKDIR /

RUN pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install supervisely==6.72.3