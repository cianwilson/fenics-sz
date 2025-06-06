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
    curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt install nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig

# BUILD VTK
# ENV VTK_VERSION="v9.2.6"
# RUN git clone --recursive --branch ${VTK_VERSION} --single-branch https://gitlab.kitware.com/vtk/vtk.git vtk && \
#     cmake -G Ninja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON vtk/ && \
#     ninja && \
#     python3 setup.py bdist_wheela

RUN echo ${TARGETPLATFORM}
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.3.0-cp312/vtk-9.3.0.dev0-cp312-cp312-linux_aarch64.whl"; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then python3 -m pip install vtk; fi

ADD pyproject.toml /tmp/pyproject.toml
RUN python3 -m pip install --no-cache-dir --no-binary=h5py -v .
RUN python3 -m pip cache purge
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
