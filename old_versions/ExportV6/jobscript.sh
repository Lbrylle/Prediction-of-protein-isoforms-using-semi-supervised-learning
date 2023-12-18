#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua10
### -- set the job Name --
#BSUB -J DLNiLu
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 8GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s203832@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o "/zhome/31/1/155455/DeepLearningProject23/output/gpu_%J.out"
#BSUB -e "/zhome/31/1/155455/DeepLearningProject23/output/gpu_%J.err"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.1
module load python3/3.10.12
source .venv/bin/activate

cd /zhome/31/1/155455/DeepLearningProject23
python3 VAE_v5_GPU.py