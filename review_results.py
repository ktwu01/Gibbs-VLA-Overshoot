#!/usr/bin/env python3
"""Review experiment results and update exp_record.md."""
import json
import os
import glob
import sys
from datetime import datetime

OUTPUTS_DIR = os.path.expanduser("~/Gibbs-VLA-Overshoot/experiments/outputs")
RECORD_FILE = os.path.expanduser("~/Gibbs-VLA-Overshoot/experiments/exp_record.md")
GIBBS_TARGET = 8.95  # theoretical Gibbs constant %

def load_results():
    """Load all experiment JSON results."""
    results = {}
    for f in sorted(glob.glob(os.path.join(OUTPUTS_DIR, "exp*.json"))):
        name = os.path.basename(f).replace(".json", "")
        try:
            with open(f) as fh:
                results[name] = json.load(fh)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def summarize(results):
    """Print summary and identify best candidates for Gibbs reproduction."""
    print(f"\n{'='*60}")
    print(f"  Gibbs Overshoot Review — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if not results:
        print("No results found yet. Experiments may still be running.")
        return

    best_name = None
    best_overshoot = 0.0
    best_per_axis = {}

    for name, data in sorted(results.items()):
        if "error" in data:
            print(f"  {name}: ERROR — {data['error']}")
            continue

        overshoot = data.get("overshoot_percent", 0.0)
        plaw = data.get("power_law_exponent", None)
        peak_v = data.get("peak_velocity", None)
        per_axis = data.get("per_axis_overshoot_percent", {})

        # Find max per-axis overshoot
        max_axis_name = ""
        max_axis_val = 0.0
        for ax, val in per_axis.items():
            if val > max_axis_val:
                max_axis_val = val
                max_axis_name = ax

        print(f"  {name}:")
        print(f"    Overshoot (analysis axis): {overshoot:.4f}%")
        print(f"    Max per-axis overshoot:    {max_axis_val:.4f}% ({max_axis_name})")
        print(f"    Power law exponent:        {plaw:.4f}" if plaw else "    Power law exponent:        N/A")
        print(f"    Peak velocity:             {peak_v:.6f}" if peak_v else "    Peak velocity:             N/A")

        instr_a = data.get("instruction_a", "?")
        instr_b = data.get("instruction_b", "?")
        print(f"    Instructions: '{instr_a}' → '{instr_b}'")
        print()

        if max_axis_val > best_overshoot:
            best_overshoot = max_axis_val
            best_name = name
            best_per_axis = per_axis

    print(f"{'='*60}")
    if best_name:
        print(f"  BEST: {best_name} — max per-axis overshoot: {best_overshoot:.4f}%")
        closeness = abs(best_overshoot - GIBBS_TARGET)
        if closeness < 1.0:
            print(f"  ** CLOSE TO GIBBS TARGET ({GIBBS_TARGET}%) — within {closeness:.2f}% **")
        elif best_overshoot > 0.1:
            print(f"  Some overshoot detected but far from target ({GIBBS_TARGET}%)")
            print(f"  Distance: {closeness:.2f}%")
        else:
            print(f"  No significant overshoot detected across any experiment.")
            print(f"  Consider: different observation frames, real Bridge V2 data,")
            print(f"            or testing the hypothesis that discrete token-based")
            print(f"            VLAs may not exhibit Gibbs-like ringing.")
    print(f"{'='*60}\n")

    return results


def update_record(results):
    """Update exp_record.md with latest results."""
    if not os.path.exists(RECORD_FILE):
        return

    with open(RECORD_FILE, "r") as f:
        content = f.read()

    for name, data in results.items():
        if "error" in data:
            continue
        # Extract exp ID (e.g., exp01 from exp01_dz_bridge_150)
        exp_id = name.split("_")[0]
        overshoot = data.get("overshoot_percent", 0.0)
        plaw = data.get("power_law_exponent", 0.0)
        peak_v = data.get("peak_velocity", 0.0)

        # Update the results table row
        old_row = f"| {exp_id} | - | - | - | PENDING |"
        new_row = f"| {exp_id} | {overshoot:.4f}% | {plaw:.4f} | {peak_v:.6f} | DONE |"
        content = content.replace(old_row, new_row)

    with open(RECORD_FILE, "w") as f:
        f.write(content)

    print(f"Updated {RECORD_FILE}")


if __name__ == "__main__":
    results = load_results()
    summarize(results)
    update_record(results)
