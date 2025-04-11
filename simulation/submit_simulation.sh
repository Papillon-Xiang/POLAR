#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --time=5:00:00
#SBATCH --array=1-75
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12


p_values=(0.95 0.75 0.55)
n_values=(50 200 1000 5000 20000)
c_values=(0 5 10 50 100)

p_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 25 % 3 + 1 ))
n_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 5 % 5 + 1 ))
c_index=$(( ($SLURM_ARRAY_TASK_ID - 1) % 5 + 1 ))

param_p=${p_values[$p_index - 1]}
param_n=${n_values[$n_index - 1]}
param_c=${c_values[$c_index - 1]}

python simulation.py --p $param_p --n $param_n --c $param_c