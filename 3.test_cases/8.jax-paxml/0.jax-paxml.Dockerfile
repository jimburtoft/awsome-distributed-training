FROM ghcr.io/nvidia/pax:nightly-2023-07-25

ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_HOME=/usr/local/cuda-12
ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH

RUN apt-get update -y && apt-get install -y pciutils tcl environment-modules && apt-get clean
RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
    libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 \
    libnccl2 libnccl-dev \
    libibumad3
RUN apt-get install -y \
 automake \
 bash \
 build-essential \
 ca-certificates \
 curl \
 debianutils \
 git \
 libtool \
 netcat \
 openssh-client \
 openssh-server \
 openssl \
 util-linux
RUN update-ca-certificates

#################################################
ENV EFA_INSTALLER_VERSION=1.23.0
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
ARG NCCL_VERSION=v2.17.1
## Install NCCL
RUN git clone https://github.com/NVIDIA/nccl /opt/nccl \
    && cd /opt/nccl \
    && git checkout -b ${NCCL_VERSION} \
    && make -j src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

#################################################
ENV AWS_OFI_NCCL_VERSION=v1.6.0
## Install AWS-OFI-NCCL plugin
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-libfabric=/opt/amazon/efa/ \
        --with-cuda=${CUDA_HOME} \
        --with-mpi=/opt/amazon/openmpi/ \
        --with-nccl=/opt/nccl/build \
    && make -j && make install

###### FIX orbax protobuf version incompatiblity
RUN pip install orbax-checkpoint==0.2.7

#################################################
ENV NCCL_TESTS_VERSION=2.13.6
## Install NCCL-tests
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl-base.html#nccl-start-base-gdrcopy
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout v${NCCL_TESTS_VERSION} \
    && make MPI=1 \
        MPI_HOME=/opt/amazon/openmpi/ \
        CUDA_HOME=${CUDA_HOME} \
        NCCL_HOME=/opt/nccl/build


# Cleanup at the end, so we don't have to keep calling apt-get update.
RUN apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    /usr/share/man/* \
    /usr/share/doc/*