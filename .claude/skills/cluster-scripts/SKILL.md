---
name: cluster-scripts
description: Generate Slurm job scripts and submission scripts for the Snellius HPC cluster from DVC pipeline stages. Use this skill whenever the user mentions "cluster", "snellius", "slurm", "sbatch", "submit job", "run on cluster", "HPC", or wants to run experiments on a remote GPU cluster. Also trigger when the user adds new DVC stages and might need corresponding cluster scripts, or asks about job submission, array jobs, or cluster resource allocation.
---

# Generating Snellius Cluster Scripts from DVC Stages

This skill translates DVC pipeline stages (defined in `dvc.yaml`) into Slurm batch scripts for the Snellius HPC cluster. The project already has a working set of cluster scripts in `cluster/` -- read them first to stay consistent with established conventions.

## Before writing anything

1. Read `dvc.yaml` to understand which stages need cluster scripts
2. Read `params.yaml` to know which parameters each stage uses
3. Read the existing cluster scripts in `cluster/` to match the project's conventions
4. Identify the DVC stage's `cmd`, `params`, `deps`, and `outs` -- these determine the job script's command, parameter reads, and output paths

## Architecture: Two-tier orchestration

DVC defines the *logical* pipeline (what to run, with what params, producing what outputs). Cluster scripts define the *physical* execution (resource allocation, environment setup, Slurm orchestration). The cluster scripts replicate DVC's commands but replace `uv run python` with direct `python` calls (the venv has everything installed).

There are three kinds of scripts, each with a distinct role:

### 1. Job scripts (`job_<name>.sh`)

These are the actual Slurm batch scripts that run on compute nodes. Each one handles a single *environment* (like T-maze or epistemic maze) but dispatches to different stages via environment variables.

**The dispatch pattern:** Rather than writing one job script per DVC stage, group related stages into a single job script that uses a `case` statement on `STAGE_TYPE` (and optionally `ANALYSIS`, `INFERENCE_MODE`, `STRATEGY`) to select what to run. This keeps the cluster directory manageable and makes the relationship between stages explicit.

**When to use a separate job script instead:** If the stage needs fundamentally different Slurm resources (different memory, time limits, array jobs), it needs its own job script. For example, MiniGrid episodes need 64GB RAM and array jobs, while aggregation needs 1 CPU and 4GB -- these can't share a script because SBATCH directives are fixed at submission time (though they can be overridden with `sbatch` flags).

### 2. Submit scripts (`submit_<name>.sh`)

These run on the login node and orchestrate job submission. They:
- Check if outputs already exist (skip completed stages)
- Submit jobs with the right env vars via `sbatch --export`
- Wire up dependencies between jobs via `--dependency=afterok:<jobid>`
- Collect job IDs for dependency chaining

### 3. Top-level dispatcher (`submit_all.sh`)

Delegates to environment-specific submit scripts. Handles MiniGrid's array job logic inline because it's structurally different from the dispatch-based environments.

## Template: Job script

```bash
#!/usr/bin/env bash
#SBATCH --job-name=aif-<environment>
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/<environment>_%x_%j.out
#SBATCH --error=logs/<environment>_%x_%j.err

# <Brief description of what this script dispatches>
# Dispatches based on STAGE_TYPE env var:
#   <stage_type_1> -- requires <VAR1> (<values>)
#   <stage_type_2> -- requires <VAR2> (<values>)
#   ...

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

echo "Running <environment> ${STAGE_TYPE} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

case "${STAGE_TYPE:?STAGE_TYPE not set}" in
    <stage_type_1>)
        python scripts/<environment>/<script>.py \
            --<flag> "${VAR:?VAR not set}" \
            --output-dir data/<environment>/<subdir>
        ;;
    *)
        echo "ERROR: unknown STAGE_TYPE='${STAGE_TYPE}'"
        exit 1
        ;;
esac

echo "<Environment> ${STAGE_TYPE} completed at $(date)"
```

### Reading parameters from `params.yaml`

When a stage needs runtime parameters that DVC would normally interpolate via `${param.path}`, read them with this helper:

```bash
read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

# Usage:
N_EPISODES=$(read_param "['environment']['n_episodes']")
```

This approach is used instead of DVC parameter interpolation because compute nodes don't need DVC installed -- they just need `params.yaml` and `pyyaml` (which is a project dependency).

**When to use `read_param` vs. hardcoded args:** If the DVC stage's `cmd` uses `${param.path}` interpolation, the cluster script needs `read_param`. If the DVC `cmd` has hardcoded flags (like `--analysis curves`), just hardcode them in the cluster script too.

### Lightweight stages (T-maze/epistemic pattern)

DVC stages that are quick (seconds to minutes) and don't need parallelism. The DVC `cmd` maps directly to a `python` call in a `case` branch.

Pattern:
- One job script per environment with `case` dispatch
- `STAGE_TYPE` selects the category (convergence/experiment/figures)
- Additional env vars (`ANALYSIS`, `INFERENCE_MODE`, `STRATEGY`) select the specific stage
- All stages share the same Slurm resource allocation

### Heavy/parallel stages (MiniGrid pattern)

DVC stages that are expensive per-episode and benefit from parallelism. These use Slurm array jobs.

```bash
#SBATCH --array=0-99
#SBATCH --output=logs/<name>_%A_%a.out
#SBATCH --error=logs/<name>_%A_%a.err

# Each array task runs one episode
python scripts/<env>/experiment.py \
    --episode-index "$SLURM_ARRAY_TASK_ID" \
    ...
```

Note the log format difference: `%A_%a` (array job ID + task ID) instead of `%x_%j` (job name + job ID).

Array jobs typically need a separate aggregation step (`job_aggregate.sh`) that runs after all episodes complete, submitted with `--dependency=afterany:<array_job_id>`.

## Template: Submit script

```bash
#!/usr/bin/env bash
# Submit all <environment> DVC stages as individual SLURM jobs.
# <Brief description of dependency structure>
# Stages whose output files already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

JOB_SCRIPT="cluster/job_<environment>.sh"

# submit_stage <label> <env_vars> [dependency_flag]
submit_stage() {
    local label="$1"
    local env_vars="$2"
    local dep_flag="${3:-}"

    sbatch --parsable \
        --job-name="$label" \
        --export="ALL,${env_vars}" \
        ${dep_flag} \
        "$JOB_SCRIPT"
}

echo "=== <Environment> stages ==="

# --- Independent stages (parallel) ---
for variant in <variant1> <variant2> <variant3>; do
    OUTPUT="data/<environment>/${variant}/results.json"
    if [ -f "$OUTPUT" ]; then
        echo "  ${variant}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "<prefix>-${variant}" "STAGE_TYPE=<type>,VAR=${variant}")
    echo "  ${variant}: $JID"
done

# --- Dependent stages ---
# Collect job IDs, then submit downstream job with --dependency=afterok:<ids>

echo ""
echo "<Environment> jobs submitted. Monitor with: squeue -u \$USER"
```

### Naming conventions

| Thing | Pattern | Examples |
|-------|---------|----------|
| Job script | `job_<environment>.sh` | `job_tmaze.sh`, `job_epistemic.sh` |
| Array job script | `job_<mode>.sh` | `job_active.sh`, `job_marginal.sh` |
| Submit script | `submit_<environment>.sh` | `submit_tmaze.sh`, `submit_epistemic.sh` |
| Job name (convergence) | `<prefix>-conv-<analysis>` | `tm-conv-curves`, `ep-conv-lr_sweep` |
| Job name (experiment) | `<prefix>-exp-<mode>` | `tm-exp-active`, `ep-exp-planning` |
| Job name (figures) | `<prefix>-figures` | `tm-figures`, `ep-figures` |
| SBATCH job-name | `aif-<environment>` | `aif-tmaze`, `aif-epistemic` |
| Log files (dispatch) | `logs/<env>_%x_%j.{out,err}` | `logs/tmaze_tm-conv-curves_12345.out` |
| Log files (array) | `logs/<name>_%A_%a.{out,err}` | `logs/active_12345_5.out` |

### Resource allocation guidelines

| Workload type | GPU | CPUs | Memory | Time |
|---|---|---|---|---|
| Lightweight (T-maze, epistemic) | 1 A100 | 18 | 16G | 30min |
| Heavy episodes (MiniGrid) | 1 A100 | 18 | 64G | 30min-1hr |
| Convergence analysis | 1 A100 | 18 | 32G | 1hr |
| Aggregation / post-processing | 1 A100 | 1 | 4G | 10min |
| Smoke test | 1 A100 | 1 | 4G | 5min |

Start with these defaults and adjust based on actual runtime. The partition is always `gpu_a100`.

### Output caching in submit scripts

Before submitting a job, check if its output already exists:

```bash
OUTPUT="data/<environment>/<subdir>/results.json"
if [ -f "$OUTPUT" ]; then
    echo "  <stage>: skipped (${OUTPUT} exists)"
    continue
fi
```

For directory outputs (like figures), check the directory:

```bash
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "  <stage>: skipped (${OUTPUT_DIR}/ exists)"
fi
```

### Dependency chaining

Collect job IDs from independent stages, then pass them to dependent stages:

```bash
# Collect IDs
DEP_JOBS=""
for ...; do
    JID=$(submit_stage ...)
    DEP_JOBS="${DEP_JOBS:+${DEP_JOBS}:}${JID}"
done

# Submit dependent stage
submit_stage "label" "STAGE_TYPE=figures" "--dependency=afterok:${DEP_JOBS}"
```

`afterok` = run only if all dependencies succeeded. Use `afterany` for jobs that should run regardless (like aggregation after array jobs, where some episodes may fail).

### Resume-friendly array jobs (MiniGrid pattern)

For array jobs, check which episodes are missing before resubmitting:

```bash
missing=$(python scripts/<env>/find_missing_episodes.py "$output_dir" "$N_EPISODES" 2>/dev/null || echo "0-$((N_EPISODES - 1))")
if [ -z "$missing" ]; then
    echo "  all $N_EPISODES episodes complete, skipping"
    return
fi
sbatch --parsable --array="${missing}%${MAX_CONCURRENT}" "$job_script"
```

The `%${MAX_CONCURRENT}` throttle limits how many array tasks run simultaneously (courtesy to other cluster users).

## Integrating with `submit_all.sh`

After creating a new environment's job + submit scripts, add a case to `submit_all.sh`:

```bash
case "$ENV" in
    <new_env>)
        bash "$SCRIPT_DIR/submit_<new_env>.sh"
        ;;
    all)
        # Add to the 'all' case:
        bash "$SCRIPT_DIR/submit_<new_env>.sh"
        echo ""
        ;;
esac
```

## Checklist when adding a new environment

1. Read the DVC stages for the new environment in `dvc.yaml`
2. Identify which stages can share a job script (same resource needs) vs. need separate scripts
3. Write the job script with dispatch `case` for each stage type
4. Write the submit script with output caching and dependency chaining
5. Add the new environment to `submit_all.sh`
6. Update `cluster/README.md` with the new environment's stages, dependencies, and resource allocation
7. Update the smoke test if there are new modules to validate
