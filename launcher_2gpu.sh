#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --mail-type=ALL
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:2
###########################

bash
source ~/.bashrc
set -ex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate VKGRNN
echo $(date '+%Y_%m_%d_%H_%M') - $SLURM_JOB_NAME - $SLURM_JOBID - `hostname` >> ./rare_entity_machine_assignments.txt
$@
