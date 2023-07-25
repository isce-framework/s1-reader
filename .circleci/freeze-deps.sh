#!/bin/sh
docker run --network=host -v "$PWD:/mnt" -w /mnt --rm -it continuumio/miniconda3 bash -c '\
    conda config --add channels conda-forge && \
    conda config --remove channels defaults && \
    conda update -y conda && \
    conda env create --file environment.yaml && conda list --explicit > specfile.txt'
