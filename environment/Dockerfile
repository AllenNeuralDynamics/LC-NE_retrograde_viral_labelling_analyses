# hash:sha256:265e7b5f6f18c9a43343ac74f7d1afa0c3158ee63782671f4237c0e6314302fa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

# Packages the code actually imports, pinned to the versions the v1 release ran
# under. Transitive dependencies are left to the resolver (top-level pins only).
RUN pip install --no-cache-dir \
    allensdk==2.16.2 \
    dask==2023.5.0 \
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
