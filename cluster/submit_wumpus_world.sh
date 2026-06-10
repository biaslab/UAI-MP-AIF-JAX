#!/usr/bin/env bash
# Submit all Wumpus World experiment stages as individual SLURM jobs.
# Each planning method runs independently.
# Stages whose output files already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

JOB_SCRIPT="cluster/job_wumpus_world.sh"

submit_stage() {
    local label="$1"
    local env_vars="$2"
    shift 2

    sbatch --parsable \
        --job-name="$label" \
        --export="ALL,${env_vars}" \
        "$@" \
        "$JOB_SCRIPT"
}

echo "=== Wumpus World experiment stages ==="

METHODS=$(python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(' '.join(p['wumpus_world_methods'].keys()))")

for method in $METHODS; do
    OUTPUT="data/results/wumpus_world/${method}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${method}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "ww-${method}" "METHOD=${method}")
    echo "  ${method}: ${JID}"
done

echo ""
echo "Wumpus World jobs submitted. Monitor with: squeue -u \$USER"
