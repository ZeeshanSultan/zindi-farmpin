# Use a docker image as base image
#FROM nvidia/cuda:9.0-cudnn7-runtime
FROM tensorflow/tensorflow:latest-gpu-jupyter
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Declare some ARGuments
##ARG PYTHON_VERSION=3.6
##ARG CONDA_VERSION=3
##ARG CONDA_PY_VERSION=4.5.11
# Installation of some libraries / RUN some commands on the base image
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# INSTALLATION OF CONDA
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "./opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
    ##echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

#RUN pip --no-cache-dir install \
#    jupyter \
#    jupyterlab 

RUN jupyter notebook --generate-config

## Add a user for jupyter following convention
RUN useradd -ms /bin/bash jovyan

#COPY docker/dev/environment.yml /environment.yml

# Use the environment.yml to create the conda environment.
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
#RUN [ "conda", "env", "create" ]
RUN conda env create

#RUN conda install ipykernel
#RUN conda activate zindi_farmpin && python -m ipykernel install --user --name zindi_farmpin

## startup the conda environment setup
### Create a conda environment to use the gpu
#RUN conda update -n base -c defaults conda && \
#    conda create -y -n zindi_farmpin -f environment.yml -c h2oai -c conda-forge h2o4gpu-cuda9 

RUN echo "conda activate zindi_farmpin && python -m ipykernel install --user --name zindi_farmpin" > ~/.bashrc
ENV PATH /opt/conda/envs/zindi_farmpin/bin:$PATH

USER jovyan

RUN echo "conda activate zindi_farmpin" > ~/.bashrc
ENV PATH /opt/conda/envs/zindi_farmpin/bin:$PATH
#RUN conda init bash
#RUN /bin/bash -c "source /home/jovyan/.bashrc"
#RUN conda activate zindi_farmpin && python -m ipykernel install --user --name zindi_farmpin

## Configure start for jupyter lab
WORKDIR /home/jovyan/
# if rebulding notebook need to delete old config
RUN rm -rf /home/jovyan/.jupyter

## Finally run jupyter lab
CMD jupyter lab --allow-root --ip=0.0.0.0
