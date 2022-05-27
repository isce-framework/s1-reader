#!/bin/sh
docker run -v $PWD:/mnt -w /mnt --rm -it continuumio/miniconda3 bash -c 'conda install -y -c conda-forge isce3=0.6 build setuptools && conda list --explicit > specfile.txt'
