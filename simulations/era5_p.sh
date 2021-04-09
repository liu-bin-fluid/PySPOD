#!/bin/bash 

####--- For matlab job with parallel computing ---- 
#PBS -q parallel20 
#PBS -l select=1:ncpus=20:mem=50GB 
#PBS -j oe 
#PBS -N era5_p 


cd $PBS_O_WORKDIR; ## This line is needed, do not modify.
np=$(cat ${PBS_NODEFILE} | wc -l);

source /etc/profile.d/rec_modules.sh
module load miniconda
bash
. ~/.bashrc

conda activate era5_python
python ERA5_MSLP_2D.py
