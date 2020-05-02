FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update
RUN apt-get install -y python3-venv vim

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/
ADD https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5 /root/.keras/models/
ADD https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5 /root/.keras/models/
RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml ./
COPY poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry run pip install -U pip
RUN poetry install --no-root
ADD . /app
WORKDIR /app
RUN poetry run python setup.py develop
