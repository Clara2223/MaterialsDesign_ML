#!/bin/sh
#BSUB -q gpuv100
#BSUB -J SchNettransfer
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:30
#BSUB -o lib/logs/transfer_%J.out
#BSUB -e lib/logs/transfer_%J.err

source ~/2024-bifrost-dtu-project/environment/bin/activate

python transfer0.py


