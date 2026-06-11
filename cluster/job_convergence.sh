#!/usr/bin/env bash
#SBATCH --job-name=aif-convergence
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/convergence_%x_%j.out
#SBATCH --error=logs/convergence_%x_%j.err

# Convergence sweep and plotting stages.
# Dispatches based on STAGE_TYPE env var:
#   sweep -- requires ENVIRONMENT (frozen-lake|wumpus-world|rocksample|minigrid)
#   plots -- no additional env vars needed

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

echo "Running convergence ${STAGE_TYPE} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

case "${STAGE_TYPE:?STAGE_TYPE not set}" in
    sweep)
        ENV="${CONV_ENV:?CONV_ENV not set}"
        N_ITERATIONS=$(read_param "['convergence_sweep']['n_iterations']")

        case "$ENV" in
            frozen-lake)
                python run_convergence_sweep.py \
                    --environment frozen-lake \
                    --grid-size "$(read_param "['convergence_sweep']['frozen_lake']['grid_size']")" \
                    --n-configs "$(read_param "['convergence_sweep']['frozen_lake']['n_configs']")" \
                    --planning-horizon "$(read_param "['convergence_sweep']['frozen_lake']['planning_horizon']")" \
                    --hole-fraction "$(read_param "['convergence_sweep']['frozen_lake']['hole_fraction']")" \
                    --min-hamming "$(read_param "['convergence_sweep']['frozen_lake']['min_hamming']")" \
                    --base-noise "$(read_param "['convergence_sweep']['frozen_lake']['base_noise']")" \
                    --noise-range "$(read_param "['convergence_sweep']['frozen_lake']['noise_range']")" \
                    --slip-prob "$(read_param "['convergence_sweep']['frozen_lake']['slip_prob']")" \
                    --hole-penalty "$(read_param "['convergence_sweep']['frozen_lake']['hole_penalty']")" \
                    --goal-temperature "$(read_param "['convergence_sweep']['frozen_lake']['goal_temperature']")" \
                    --scan-cost "$(read_param "['convergence_sweep']['frozen_lake']['scan_cost']")" \
                    --n-iterations "$N_ITERATIONS" \
                    --output-dir data/convergence_sweep
                ;;
            wumpus-world)
                python run_convergence_sweep.py \
                    --environment wumpus-world \
                    --grid-size "$(read_param "['convergence_sweep']['wumpus_world']['grid_size']")" \
                    --n-configs "$(read_param "['convergence_sweep']['wumpus_world']['n_configs']")" \
                    --n-pits "$(read_param "['convergence_sweep']['wumpus_world']['n_pits']")" \
                    --planning-horizon "$(read_param "['convergence_sweep']['wumpus_world']['planning_horizon']")" \
                    --obs-noise "$(read_param "['convergence_sweep']['wumpus_world']['obs_noise']")" \
                    --pos-noise "$(read_param "['convergence_sweep']['wumpus_world']['pos_noise']")" \
                    --slip-prob "$(read_param "['convergence_sweep']['wumpus_world']['slip_prob']")" \
                    --pit-penalty "$(read_param "['convergence_sweep']['wumpus_world']['pit_penalty']")" \
                    --wumpus-penalty "$(read_param "['convergence_sweep']['wumpus_world']['wumpus_penalty']")" \
                    --goal-temperature "$(read_param "['convergence_sweep']['wumpus_world']['goal_temperature']")" \
                    --sense-cost "$(read_param "['convergence_sweep']['wumpus_world']['sense_cost']")" \
                    --n-iterations "$N_ITERATIONS" \
                    --output-dir data/convergence_sweep
                ;;
            rocksample)
                python run_convergence_sweep.py \
                    --environment rocksample \
                    --grid-size "$(read_param "['convergence_sweep']['rocksample']['grid_size']")" \
                    --n-rocks "$(read_param "['convergence_sweep']['rocksample']['n_rocks']")" \
                    --planning-horizon "$(read_param "['convergence_sweep']['rocksample']['planning_horizon']")" \
                    --half-eff-dist "$(read_param "['convergence_sweep']['rocksample']['half_eff_dist']")" \
                    --pos-noise "$(read_param "['convergence_sweep']['rocksample']['pos_noise']")" \
                    --slip-prob "$(read_param "['convergence_sweep']['rocksample']['slip_prob']")" \
                    --good-logit "$(read_param "['convergence_sweep']['rocksample']['good_logit']")" \
                    --bad-logit "$(read_param "['convergence_sweep']['rocksample']['bad_logit']")" \
                    --exit-logit "$(read_param "['convergence_sweep']['rocksample']['exit_logit']")" \
                    --goal-temperature "$(read_param "['convergence_sweep']['rocksample']['goal_temperature']")" \
                    --sense-cost "$(read_param "['convergence_sweep']['rocksample']['sense_cost']")" \
                    --sample-cost "$(read_param "['convergence_sweep']['rocksample']['sample_cost']")" \
                    --terminal-goal-only \
                    --n-iterations "$N_ITERATIONS" \
                    --output-dir data/convergence_sweep
                ;;
            minigrid)
                python run_convergence_sweep.py \
                    --environment minigrid \
                    --grid-size "$(read_param "['convergence_sweep']['minigrid']['grid_size']")" \
                    --planning-horizon "$(read_param "['convergence_sweep']['minigrid']['planning_horizon']")" \
                    --fov-size "$(read_param "['convergence_sweep']['minigrid']['fov_size']")" \
                    --obs-alpha "$(read_param "['convergence_sweep']['minigrid']['obs_alpha']")" \
                    --observe-first \
                    --n-iterations "$N_ITERATIONS" \
                    --output-dir data/convergence_sweep
                ;;
            *)
                echo "ERROR: unknown ENVIRONMENT='${ENV}'"
                exit 1
                ;;
        esac
        ;;
    plots)
        python plot_convergence_sweep.py \
            --environment all \
            --input-dir data/convergence_sweep \
            --output-dir data/convergence_sweep/plots \
            --format both \
            --tex
        ;;
    *)
        echo "ERROR: unknown STAGE_TYPE='${STAGE_TYPE}'"
        exit 1
        ;;
esac

echo "Convergence ${STAGE_TYPE} completed at $(date)"
