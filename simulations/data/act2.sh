#!/bin/bash

#PBS -P era5_download_
#PBS -j oe
#PBS -N era5
#PBS -q openmp
#PBS -l select=1:ncpus=1:mem=20GB

cd $PBS_O_WORKDIR; ## This line is needed, do not modify.
np=$(cat ${PBS_NODEFILE} | wc -l);

source /etc/profile.d/rec_modules.sh
module load miniconda
bash
. ~/.bashrc

conda activate era5_python
python summer.py
 
###python ENSO_MEI_2D_trial.py
