"""Run every CPU mid-term step for one scene, in order (laptop-friendly, ~10-30 min total).

  1 inspect_scene     facts + pairs/cameras/sharpness figures
  2 blur_maps         blur-length map, measured blur vs depth
  3 make_strata       curvature/proximity strata on the laser scan (+ coloured PLYs)
  4 validate_harness  checks V1-V3 (the geometry scoring is trustworthy)
  5 eval_published    per-stratum geometry of RealX3D's six published methods (needs baseline_results)

Usage:  python scripts/run_midterm_cpu.py --data data --scene Ujikintoki [--skip 5]
"""
import argparse
import os
import subprocess
import sys
import time

STEPS = [("1", "inspect_scene.py", []), ("2", "blur_maps.py", []), ("3", "make_strata.py", []),
         ("4", "validate_harness.py", []), ("5", "eval_published.py", [])]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default="data")
ap.add_argument("--scene", default="Ujikintoki")
ap.add_argument("--out", default="outputs")
ap.add_argument("--skip", nargs="*", default=[])
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
for num, script, extra in STEPS:
    if num in a.skip:
        continue
    t = time.time()
    print(f"\n===== step {num}: {script} =====", flush=True)
    r = subprocess.run([sys.executable, os.path.join(here, script), "--data", a.data, "--scene", a.scene,
                        "--out", a.out] + extra)
    print(f"===== step {num} {'OK' if r.returncode == 0 else 'FAILED (code %d)' % r.returncode} "
          f"in {time.time() - t:.0f}s =====", flush=True)
print(f"\nAll outputs are in {os.path.join(a.out, a.scene)}")
