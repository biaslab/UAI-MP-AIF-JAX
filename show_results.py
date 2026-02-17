#!/usr/bin/env python3
"""Print a performance comparison table from data/results/*.json."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "data" / "results"

ORDER = ["bp", "loopy", "reduced-aif", "region-extended", "reduced-nuijten", "nuijten"]


def load_results():
    rows = {}
    for path in RESULTS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        method = data["config"]["planning_method"]
        rows[method] = {
            "plan_iters": data["config"]["planning_iterations"],
            "success_rate": data["results"]["success_rate"],
            "avg_steps": data["results"]["avg_steps"],
            "avg_reward": data["results"]["avg_reward"],
            "total_time_s": data["results"]["total_time_s"],
        }
    return rows


def print_table(rows):
    # Sort by ORDER list, then any remaining methods by success rate descending
    known = [m for m in ORDER if m in rows]
    extra = sorted(
        [m for m in rows if m not in ORDER],
        key=lambda m: rows[m]["success_rate"],
        reverse=True,
    )
    methods = known + extra

    col_widths = {
        "method": max(len("Method"), max(len(m) for m in methods)),
        "plan_iters": len("Plan Iters"),
        "success_rate": len("Success Rate"),
        "avg_steps": len("Avg Steps"),
        "avg_reward": len("Avg Reward"),
        "total_time_s": len("Total Time (s)"),
    }

    header = (
        f"{'Method':<{col_widths['method']}} | "
        f"{'Plan Iters':>{col_widths['plan_iters']}} | "
        f"{'Success Rate':>{col_widths['success_rate']}} | "
        f"{'Avg Steps':>{col_widths['avg_steps']}} | "
        f"{'Avg Reward':>{col_widths['avg_reward']}} | "
        f"{'Total Time (s)':>{col_widths['total_time_s']}}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for method in methods:
        r = rows[method]
        print(
            f"{method:<{col_widths['method']}} | "
            f"{r['plan_iters']:>{col_widths['plan_iters']}} | "
            f"{r['success_rate']*100:>{col_widths['success_rate'] - 1}.0f}% | "
            f"{r['avg_steps']:>{col_widths['avg_steps']}.2f} | "
            f"{r['avg_reward']:>{col_widths['avg_reward']}.4f} | "
            f"{r['total_time_s']:>{col_widths['total_time_s']}.1f}"
        )

    print(sep)


if __name__ == "__main__":
    rows = load_results()
    print_table(rows)
