"""Train + score several runs one after another (for Colab/Kaggle). Skips runs that are already done.

Each run spec is  condition:images:steps[:tag]
  images = blurred | sharp | mixed:0.25
Examples:
  python scripts/run_batch.py --scene Ujikintoki --runs motion_strong:sharp:7000:sharp_control \
         motion_mild:blurred:7000 defocus_mild:blurred:7000
  python scripts/run_batch.py --runs motion_strong:blurred:30000:vanilla_30k

Every run ends with  runs/<Scene>/<condition>/<tag>/s0/{metrics.json, geometry.json, previews/ ...}
and a line in runs/batch_log.txt. If a run crashes, the batch continues with the next one.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--backup", default=None,
                    help="folder (e.g. on Google Drive) where each finished run's small files are copied; "
                         "runs already there are skipped, so a fresh Colab machine can resume")
    ap.add_argument("--train-args", default="", help="extra arguments for train_gsplat.py, e.g. \"--device cpu --downscale 16\"")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "batch_log.txt"), "a")
    for spec in a.runs:
        parts = spec.split(":")
        if len(parts) < 3:
            print(f"bad run spec '{spec}' (expected condition:images:steps[:tag]); skipping")
            continue
        if parts[1] == "mixed":                      # mixed:0.25 contains a colon
            parts = [parts[0], parts[1] + ":" + parts[2]] + parts[3:]
        cond, images, steps = parts[0], parts[1], int(parts[2])
        tag = parts[3] if len(parts) > 3 else {"blurred": "vanilla", "sharp": "sharp_control"}.get(
            images, images.replace(":", ""))
        run_dir = os.path.join(a.out, a.scene, cond, tag, "s0")
        t0 = time.time()
        print(f"\n########## {spec}  ->  {run_dir}", flush=True)
        bdir = os.path.join(a.backup, a.scene, cond, tag, "s0") if a.backup else None
        if os.path.exists(os.path.join(run_dir, "geometry.json")) or \
                (bdir and os.path.exists(os.path.join(bdir, "geometry.json"))):
            print("already done, skipping")
            continue
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            r = subprocess.run([sys.executable, os.path.join(here, "train_gsplat.py"), "--data", a.data,
                                "--scene", a.scene, "--condition", cond, "--images", images,
                                "--steps", str(steps), "--tag", tag, "--out", a.out] + a.train_args.split())
            if r.returncode != 0:
                msg = f"{spec}: TRAINING FAILED (code {r.returncode})"
                print(msg)
                log.write(msg + "\n")
                log.flush()
                continue
        r = subprocess.run([sys.executable, os.path.join(here, "eval_run.py"), "--run", run_dir,
                            "--data", a.data])
        if r.returncode == 0 and bdir:
            shutil.copytree(run_dir, bdir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("depth", "*.ply", "*.npz"))
            print(f"copied small files to {bdir}")
        msg = f"{spec}: {'ok' if r.returncode == 0 else 'EVAL FAILED'} in {(time.time() - t0) / 60:.1f} min"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()


if __name__ == "__main__":
    main()
