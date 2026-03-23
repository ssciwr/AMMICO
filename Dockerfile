FROM quay.io/jupyter/base-notebook:python-3.13

# Install system dependencies for audio decoding
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER
# Copy the repository into the container
COPY --chown=${NB_UID} . /opt/ammico

# Install the Python dependencies
RUN pip install uv && uv pip install --system --no-cache-dir "/opt/ammico[nb]"

# Make JupyterLab the default for this application
ENV JUPYTER_ENABLE_LAB=yes

# copy the notebooks into the container
COPY --chown=${NB_UID} ./docs/tutorials/*.ipynb /home/jovyan/notebooks/.
# Copy sample data into the container
COPY --chown=${NB_UID} ./data/in/*.jpeg /opt/ammico/data

# Export where the data is located
ENV XDG_DATA_HOME=/opt/ammico/data
  