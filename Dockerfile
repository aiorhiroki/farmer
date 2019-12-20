FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y vim git locales \
    libglib2.0-0 libsm6 libxrender1 libxext6 

RUN pip install --upgrade pip

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY Pipfile ./
COPY Pipfile.lock ./
RUN pip install pipenv
RUN pipenv install --system
RUN rm Pipfile.lock

RUN pip install git+https://github.com/aiorhiroki/farmer.git@v1.1.1
RUN pip install git+https://github.com/aiorhiroki/ncc.git@v1.0.0

ADD "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5" /root/.keras/models/
ADD "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5" /root/.keras/models/

WORKDIR /home

