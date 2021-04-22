FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update
RUN apt-get install -y python3-venv vim libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml ./
COPY poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-dev
