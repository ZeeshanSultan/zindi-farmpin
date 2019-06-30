FROM ubuntu:18.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y \
    byobu \
    ca-certificates \
    curl \
    git-core \
    htop \
    unzip \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH /opt/conda/bin:$PATH



RUN conda install tensorflow -y

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
    awscli \
    keras \
    numpy \
    pandas \
    psycopg2 \
    sqlalchemy \
    sqlalchemy_redshift

RUN mkdir -p /home/work/

WORKDIR /home/work/

CMD python /home/work/src/start.py \
    && /bin/bash # For now (comment out for actual deployment via PiedPiper so that the process can end).
