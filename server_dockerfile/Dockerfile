FROM python:3.8-slim
LABEL maintainer="Moontrey"

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN pip install mlflow

# EXPOSE ${SERVER_PORT} - moved to docker-compose

# WORKDIR ${MLFLOW_HOME}

USER user

# CMD mlflow server --host 0.0.0.0:5000 - moved to docker-compose
