#!/bin/bash

#SBATCH --job-name=mod_arch
#SBATCH --output=job.out
#SBATCH --error=job.out
#SBATCH --open-mode=append
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=10:00:00             # Time limit hrs:min:sec

#conda activate modti

#cd /home/roy/MLSB2021_PLM_DTI/modti/apps
#python train.py configs/modular.yaml

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID
eval "$(conda shell.bash hook)"

echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"

let seed=SLURM_ARRAY_TASK_ID+1*22
echo $seed

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# Run the loop of runs for this task.
declare -a StringArray=('attn' 'mil-attn' 'mil-gated' 'gated' 'max' 'mean' 'avg' 'sum')
for value in ${StringArray[@]};
do
    echo $value
    conda activate modti
    cd /home/roy/MLSB2021_PLM_DTI/modti/apps
    python train.py configs/modular.yaml seed=$seed model.feat_agg=$value model.pred_agg='mil-attn'
done

date