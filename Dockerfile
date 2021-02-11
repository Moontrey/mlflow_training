FROM python:3.8
LABEL maintainer="Moontrey"

ENV MLFLOW_HOME /opt/mlflow
ENV MLFLOW_VERSION 1.13.1
ENV FILE_STORE ${MLFLOW_HOME}/fileStore

ENV ARTIFACT_STORE ${MLFLOW_HOME}/artifactStore
ENV SERVER_HOST 0.0.0.0
ENV SERVER_PORT 5000

RUN pip install mlflow==${MLFLOW_VERSION} && \
     mkdir -p ${MLFLOW_HOME}/scripts && \
     mkdir -p ${FILE_STORE} && \
     mkdir -p ${ARTIFACT_STORE}

COPY run.sh ${MLFLOW_HOME}/scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/scripts/run.sh

EXPOSE ${SERVER_PORT}

VOLUME ["${MLFLOW_HOME}/scripts/", "${FILE_STORE}", "${ARTIFACT_STORE}"]

WORKDIR ${MLFLOW_HOME}

# ENTRYPOINT ["./scripts/run.sh"]
CMD mlflow server --host 0.0.0.0:5000