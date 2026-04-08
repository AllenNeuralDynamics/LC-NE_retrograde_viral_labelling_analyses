# hash:sha256:265e7b5f6f18c9a43343ac74f7d1afa0c3158ee63782671f4237c0e6314302fa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASK-PASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    antspyx==0.4.2 \
    dask==2023.5.0 \
    opencv-python-headless==4.8.0.74 \
    zarr==2.14.2

RUN pip install -U --no-cache-dir \
    pandas \
    scikit-learn \
    seaborn \
    rich \
    tqdm \
    timebudget \
    ruff \
    k3d \
    trimesh \
    rtree \
    openpyxl \
    matplotlib-venn

COPY postInstall /
RUN /postInstall