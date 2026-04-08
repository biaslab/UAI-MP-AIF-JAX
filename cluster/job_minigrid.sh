#!/usr/bin/env bash
#SBATCH --job-name=aif-minigrid
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/minigrid_%x_%j.out
#SBATCH --error=logs/minigrid_%x_%j.err

# MiniGrid experiment stage.
# Requires METHOD env var (loopy|dyn-channel|nuijten|vbp-channel|active-inference)

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

METHOD="${METHOD:?METHOD not set}"
RECEDING=$(read_param "['minigrid']['receding_horizon']")
RECORD=$(read_param "['minigrid']['record']")

echo "Running minigrid method=${METHOD} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python run_minigrid.py \
    --planning-method "$METHOD" \
    --grid-size "$(read_param "['minigrid']['grid_size']")" \
    --fov-size "$(read_param "['minigrid']['fov_size']")" \
    --obs-alpha "$(read_param "['minigrid']['obs_alpha']")" \
    --episodes "$(read_param "['minigrid']['episodes']")" \
    --max-steps "$(read_param "['minigrid']['max_steps']")" \
    --planning-horizon "$(read_param "['minigrid']['planning_horizon']")" \
    --inference-iterations "$(read_param "['minigrid']['inference_iterations']")" \
    --planning-iterations "$(read_param "['minigrid_methods']['$METHOD']['planning_iterations']")" \
    --damping "$(read_param "['minigrid_methods']['$METHOD']['damping']")" \
    --seed "$(read_param "['minigrid']['seed']")" \
    $([ "$RECEDING" = "True" ] && echo "--receding-horizon") \
    --output "data/results/minigrid/${METHOD}.json" \
    --record "$RECORD"

echo "MiniGrid method=${METHOD} completed at $(date)"
