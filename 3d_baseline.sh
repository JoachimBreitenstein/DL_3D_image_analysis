#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 06:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o models/3d_segmentation_%J.out
#BSUB -e models/3d_segmentation_%J.err
# -- end of LSF options --

source /dtu/3d-imaging-center/courses/conda/conda_init.sh
conda activate env-02510
python -u 3d_baseline_final.py