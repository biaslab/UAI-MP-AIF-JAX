#!/usr/bin/env bash
#SBATCH --job-name=aif-frozen-lake
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/frozen_lake_%x_%j.out
#SBATCH --error=logs/frozen_lake_%x_%j.err

# Frozen Lake experiment stage.
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

echo "Running frozen_lake method=${METHOD} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python run_frozen_lake.py \
    --planning-method "$METHOD" \
    --grid-size "$(read_param "['frozen_lake']['grid_size']")" \
    --n-configs "$(read_param "['frozen_lake']['n_configs']")" \
    --hole-fraction "$(read_param "['frozen_lake']['hole_fraction']")" \
    --min-hamming "$(read_param "['frozen_lake']['min_hamming']")" \
    --obs-noise "$(read_param "['frozen_lake']['obs_noise']")" \
    --slip-prob "$(read_param "['frozen_lake']['slip_prob']")" \
    --episodes "$(read_param "['frozen_lake']['episodes']")" \
    --max-steps "$(read_param "['frozen_lake']['max_steps']")" \
    --planning-horizon "$(read_param "['frozen_lake']['planning_horizon']")" \
    --planning-iterations "$(read_param "['frozen_lake_methods']['$METHOD']['planning_iterations']")" \
    --damping "$(read_param "['frozen_lake_methods']['$METHOD']['damping']")" \
    --hole-penalty "$(read_param "['frozen_lake']['hole_penalty']")" \
    --goal-temperature "$(read_param "['frozen_lake']['goal_temperature']")" \
    --seed "$(read_param "['frozen_lake']['seed']")" \
    --receding-horizon \
    --output "data/results/frozen_lake/${METHOD}.json"

echo "Frozen Lake method=${METHOD} completed at $(date)"
