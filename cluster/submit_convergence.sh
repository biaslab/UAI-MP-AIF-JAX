#!/usr/bin/env bash
# Submit all convergence sweep DVC stages as individual SLURM jobs.
# 4 environment sweeps run in parallel, then plots depend on all 4.
# Stages whose output files already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

JOB_SCRIPT="cluster/job_convergence.sh"

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

echo "=== Convergence sweep stages ==="

DEP_JOBS=""

for env in frozen-lake wumpus-world rocksample minigrid; do
    OUTPUT="data/convergence_sweep/${env}/summary.csv"
    if [ -f "$OUTPUT" ]; then
        echo "  ${env}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "conv-${env}" "STAGE_TYPE=sweep,CONV_ENV=${env}")
    DEP_JOBS="${DEP_JOBS:+${DEP_JOBS}:}${JID}"
    echo "  ${env}: ${JID}"
done

# --- Convergence plots (depends on all sweeps) ---
OUTPUT_DIR="data/convergence_sweep/plots"
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "  plots: skipped (${OUTPUT_DIR}/ exists)"
else
    if [ -n "$DEP_JOBS" ]; then
        JID=$(submit_stage "conv-plots" "STAGE_TYPE=plots" \
            "--dependency=afterok:${DEP_JOBS}" \
            --mem=4G --cpus-per-task=1 --time=00:10:00)
    else
        JID=$(submit_stage "conv-plots" "STAGE_TYPE=plots" \
            --mem=4G --cpus-per-task=1 --time=00:10:00)
    fi
    echo "  plots: ${JID}"
fi

echo ""
echo "Convergence jobs submitted. Monitor with: squeue -u \$USER"
