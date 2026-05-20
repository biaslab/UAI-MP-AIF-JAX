#!/usr/bin/env bash
#SBATCH --job-name=aif-wumpus-world
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wumpus_world_%x_%j.out
#SBATCH --error=logs/wumpus_world_%x_%j.err

# Wumpus World experiment stage.
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

echo "Running wumpus_world method=${METHOD} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python run_wumpus_world.py \
    --planning-method "$METHOD" \
    --grid-size "$(read_param "['wumpus_world']['grid_size']")" \
    --n-configs "$(read_param "['wumpus_world']['n_configs']")" \
    --n-pits "$(read_param "['wumpus_world']['n_pits']")" \
    --obs-noise "$(read_param "['wumpus_world']['obs_noise']")" \
    --pos-noise "$(read_param "['wumpus_world']['pos_noise']")" \
    --slip-prob "$(read_param "['wumpus_world']['slip_prob']")" \
    --episodes "$(read_param "['wumpus_world']['episodes']")" \
    --max-steps "$(read_param "['wumpus_world']['max_steps']")" \
    --planning-horizon "$(read_param "['wumpus_world']['planning_horizon']")" \
    --planning-iterations "$(read_param "['wumpus_world_methods']['$METHOD']['planning_iterations']")" \
    --damping "$(read_param "['wumpus_world_methods']['$METHOD']['damping']")" \
    --scan-cost "$(read_param "['wumpus_world']['scan_cost']")" \
    --pit-penalty "$(read_param "['wumpus_world']['pit_penalty']")" \
    --wumpus-penalty "$(read_param "['wumpus_world']['wumpus_penalty']")" \
    --goal-temperature "$(read_param "['wumpus_world']['goal_temperature']")" \
    --seed "$(read_param "['wumpus_world']['seed']")" \
    --receding-horizon \
    --output "data/results/wumpus_world/${METHOD}.json" \
    --record-trajectories "$(read_param "['wumpus_world']['record_trajectories']")" \
    --trajectory-dir "data/trajectories/wumpus_world/${METHOD}"

echo "Wumpus World method=${METHOD} completed at $(date)"
