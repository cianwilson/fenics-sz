FROM ghcr.io/fenics/dolfinx/lab:v0.9.0r1
ARG TARGETPLATFORM


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="static"
ENV OMP_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV XDG_RUNTIME_DIR='/tmp'

WORKDIR /tmp/
# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev libxrender1 xvfb curl vim ack zip gh gmt libgmt-dev ghostscript && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    # curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh && \
    # bash nodesource_setup.sh && \
    # apt install nodejs && \

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig && \
    python3 -m pip cache purge

# BUILD VTK
# ENV VTK_VERSION="v9.2.6"
# RUN git clone --recursive --branch ${VTK_VERSION} --single-branch https://gitlab.kitware.com/vtk/vtk.git vtk && \
#     cmake -G Ninja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON vtk/ && \
#     ninja && \
#     python3 setup.py bdist_wheela

RUN echo ${TARGETPLATFORM}
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install --no-cache-dir "https://github.com/scientificcomputing/vtk-aarch64/releases/download/v9.4.2-py3.12-more-flags/vtk-9.4.2-cp312-cp312-linux_aarch64.whl"; pip install --no-cache-dir pyvista[trame]==0.47.3; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then python3 -m pip install --no-cache-dir vtk==9.4.2; pip install --no-cache-dir pyvista[trame]==0.47.3; fi

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-binary=h5py --no-build-isolation -v h5py==3.13.0 -r requirements.txt && \
    python3 -m pip cache purge
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
