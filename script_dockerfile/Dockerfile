FROM python:3.8-slim
LABEL maintainer="Moontrey"

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN pip install --upgrade cython pip

COPY ./requirements.txt .
RUN pip install -r requirements.txt

# COPY train.py ${MODEL_HOME}/scripts/train.py - made by volume in compose
# COPY data_processed.csv ${MODEL_HOME}/data/data_processed.csv - made by volume in compose

ENV GIT_PYTHON_REFRESH=quiet
USER user

# CMD python opt/model/scripts/train.py - mooved to docker-compose 
