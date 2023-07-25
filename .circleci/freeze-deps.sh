#!/bin/sh
docker run -v "$PWD:/mnt" -w /mnt --rm -it continuumio/miniconda3 bash -c '\
    conda config --add channels conda-forge && \
    conda config --remove channels defaults && \
    conda update -y conda && \
    conda install -y --file environment.yaml && conda list --explicit > specfile.txt'
