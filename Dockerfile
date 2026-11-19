FROM quay.io/jupyter/base-notebook:python-3.13

# Install system dependencies for audio decoding
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv

USER $NB_USER
# Install the dependencies first, from pyproject.toml only: this layer is cached
# and only rebuilt when pyproject.toml changes, not on every code/notebook change.
# The cpu extra installs CPU-only torch (no CUDA libraries), see pyproject.toml
COPY --chown=${NB_UID} pyproject.toml /opt/ammico/pyproject.toml
RUN uv pip install --system --no-cache -r /opt/ammico/pyproject.toml \
    --extra cpu --extra nb --extra api

# .git is not part of the build context, so setuptools-scm cannot infer the
# version; it is passed in from the workflow instead
ARG AMMICO_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMMICO=${AMMICO_VERSION}

# Copy the repository and install ammico itself (dependencies are already there)
COPY --chown=${NB_UID} . /opt/ammico
RUN uv pip install --system --no-cache --no-deps /opt/ammico

# Make JupyterLab the default for this application
ENV JUPYTER_ENABLE_LAB=yes

# copy the notebooks into the container
COPY --chown=${NB_UID} ./docs/tutorials/*.ipynb /home/jovyan/notebooks/.
# copy the workshop files into the container
COPY --chown=${NB_UID} ./docs/workshop/* /home/jovyan/workshop/.
# Copy sample data into the container
# COPY --chown=${NB_UID} ./data/in/*.jpeg /opt/ammico/data

# Export where the data is located
ENV AMMICO_DATA_HOME=/opt/ammico/data
  