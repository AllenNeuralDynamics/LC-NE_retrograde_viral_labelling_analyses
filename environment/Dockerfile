# hash:sha256:265e7b5f6f18c9a43343ac74f7d1afa0c3158ee63782671f4237c0e6314302fa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

# Packages the code imports, pinned to specific versions. Transitive
# dependencies are left to the resolver (top-level pins only). The CCF region
# lookup uses brainglobe-atlasapi, reading the mounted .brainglobe atlas.
RUN pip install --no-cache-dir \
    brainglobe-atlasapi==2.1.0 \
    ipywidgets==8.0.7 \
    k3d==2.17.0 \
    matplotlib==3.7.2 \
    numpy==1.23.5 \
    pandas==2.3.3 \
    plotly==6.6.0 \
    rtree==1.4.1 \
    ruff==0.15.9 \
    scikit-learn==1.7.2 \
    scipy==1.10.1 \
    seaborn==0.13.2 \
    trimesh==4.11.5

COPY postInstall /
RUN /postInstall
