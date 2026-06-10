#!/usr/bin/env bash
#SBATCH --job-name=aif-rocksample
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rocksample_%x_%j.out
#SBATCH --error=logs/rocksample_%x_%j.err

# RockSample experiment stage.
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

echo "Running rocksample method=${METHOD} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python run_rocksample.py \
    --planning-method "$METHOD" \
    --grid-size "$(read_param "['rocksample']['grid_size']")" \
    --n-rocks "$(read_param "['rocksample']['n_rocks']")" \
    --half-eff-dist "$(read_param "['rocksample']['half_eff_dist']")" \
    --pos-noise "$(read_param "['rocksample']['pos_noise']")" \
    --slip-prob "$(read_param "['rocksample']['slip_prob']")" \
    --episodes "$(read_param "['rocksample']['episodes']")" \
    --max-steps "$(read_param "['rocksample']['max_steps']")" \
    --planning-horizon "$(read_param "['rocksample']['planning_horizon']")" \
    --planning-iterations "$(read_param "['rocksample_methods']['$METHOD']['planning_iterations']")" \
    --damping "$(read_param "['rocksample_methods']['$METHOD']['damping']")" \
    --good-reward "$(read_param "['rocksample']['good_reward']")" \
    --bad-penalty "$(read_param "['rocksample']['bad_penalty']")" \
    --exit-reward "$(read_param "['rocksample']['exit_reward']")" \
    --good-logit "$(read_param "['rocksample']['good_logit']")" \
    --bad-logit "$(read_param "['rocksample']['bad_logit']")" \
    --exit-logit "$(read_param "['rocksample']['exit_logit']")" \
    --goal-temperature "$(read_param "['rocksample']['goal_temperature']")" \
    --sense-cost "$(read_param "['rocksample']['sense_cost']")" \
    --sample-cost "$(read_param "['rocksample']['sample_cost']")" \
    --terminal-goal-only \
    --seed "$(read_param "['rocksample']['seed']")" \
    --receding-horizon \
    --output "data/results/rocksample/${METHOD}.json"

echo "RockSample method=${METHOD} completed at $(date)"
