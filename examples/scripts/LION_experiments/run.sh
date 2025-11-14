#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:30:00

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
conda activate verona_env


export TMP="/scratch-shared/abosman"
export TEMP="/scratch-shared/abosman"
export TMPDIR="/scratch-shared/abosman"
mkdir -p $TMPDIR

python /home/abosman/dev/VERONA/examples/scripts/LION_experiments/main.py --model_name {model_name} --attack_method {attack_method} --indices {indice} --batch_size {batch_size}
# python /home/abosman/dev/VERONA/examples/scripts/LION_experiments/test_missing.py --model_name vit_large_patch14_clip_224.openai_ft_in12k_in1k --attack_method pgd --indices 0 --batch_size 5000
