#!/bin/bash
# Helper script to submit GRPO training jobs to SLURM
# Usage: ./submit_job.sh [run_grpo.sh arguments]
# Example: ./submit_job.sh --dataset function_correctness --k -0.001 --epochs 5

# Check if logs directory exists
mkdir -p logs

# Submit the job and pass all arguments to run_grpo.sh
sbatch submit_grpo.slurm "$@"

echo "Job submitted! Use 'squeue -u $USER' to check status"
echo "Logs will be in: logs/slurm-<jobid>.out and logs/slurm-<jobid>.err"
