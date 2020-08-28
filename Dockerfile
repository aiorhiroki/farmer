FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update
RUN apt-get install -y python3-venv vim libsm6 libxrender1 libxext-dev

RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml ./
COPY poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
