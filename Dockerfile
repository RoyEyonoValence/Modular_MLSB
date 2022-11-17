FROM gcr.io/platform-iv/conda-gpu

ARG INVIVOAI_ANACONDA_USER_BOT_TOKEN

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV CONDA_ENV_NAME=modti

ADD . /app

WORKDIR /app

RUN bash --login /app/docker/bootstrap.sh $INVIVOAI_ANACONDA_USER_BOT_TOKEN

SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
