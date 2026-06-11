#!/usr/bin/env bash
# Submit all DVC pipeline stages to the Snellius cluster.
# Usage: bash cluster/submit_all.sh [convergence|frozen-lake|wumpus-world|rocksample|minigrid|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV="${1:-all}"

case "$ENV" in
    convergence)
        bash "$SCRIPT_DIR/submit_convergence.sh"
        ;;
    frozen-lake)
        bash "$SCRIPT_DIR/submit_frozen_lake.sh"
        ;;
    wumpus-world)
        bash "$SCRIPT_DIR/submit_wumpus_world.sh"
        ;;
    rocksample)
        bash "$SCRIPT_DIR/submit_rocksample.sh"
        ;;
    minigrid)
        bash "$SCRIPT_DIR/submit_minigrid.sh"
        ;;
    all)
        # MiniGrid excluded for now (not in the paper);
        # run it explicitly with: bash cluster/submit_all.sh minigrid
        bash "$SCRIPT_DIR/submit_convergence.sh"
        echo ""
        bash "$SCRIPT_DIR/submit_frozen_lake.sh"
        echo ""
        bash "$SCRIPT_DIR/submit_wumpus_world.sh"
        echo ""
        bash "$SCRIPT_DIR/submit_rocksample.sh"
        ;;
    *)
        echo "Usage: $0 [convergence|frozen-lake|wumpus-world|rocksample|minigrid|all]"
        exit 1
        ;;
esac
