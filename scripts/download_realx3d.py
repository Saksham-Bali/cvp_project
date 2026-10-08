"""Download the parts of RealX3D (Hugging Face: ToferFish/RealX3D) used by this project.

What it can fetch (all for the 4 blur conditions and the 8 blur scenes by default):
  quarter    data_4 images + poses (~0.85 GB per condition for all 8 scenes; ~0.17 GB per scene)
  gt         laser-scanned ground truth per scene (mesh, point cloud, depth; ~0.12-0.41 GB per scene)
  baselines  RealX3D's own renders (RGB + depth) of the 6 published methods (7 GB archive per
             condition; only the scenes you ask for are kept, and streaming stops early when it can)

Examples (run from the repo folder):
  python scripts/download_realx3d.py --what quarter gt --scenes Ujikintoki
  python scripts/download_realx3d.py --what all --scenes Ujikintoki --conditions motion_strong defocus_strong
  python scripts/download_realx3d.py --what quarter gt            (all 8 scenes, all 4 blur conditions)

Needs:  pip install huggingface_hub requests
"""
import argparse
import os
import shutil
import sys
import tarfile

import requests
from huggingface_hub import hf_hub_download

REPO = "ToferFish/RealX3D"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
BLUR_CONDITIONS = ["motion_mild", "motion_strong", "defocus_mild", "defocus_strong"]
# Order matches the order inside the archives, so asking for early scenes downloads less.
BLUR_SCENES = ["Ujikintoki", "Laboratory", "Popcorn", "Cupcake",
               "GearWorks", "Chocolate", "MilkCookie", "Limon"]


def safe_extract(tf, member, out_dir):
    try:
        tf.extract(member, out_dir, filter="data")  # Python 3.12+
    except TypeError:
        tf.extract(member, out_dir)


def extract_archive(archive, out_dir, keep_scenes=None):
    """Extract a .tar.gz. If keep_scenes is given, keep only paths whose 2nd component is in it."""
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            parts = m.name.split("/")
            if keep_scenes and len(parts) >= 2 and parts[1] not in keep_scenes:
                continue
            safe_extract(tf, m, out_dir)


def fetch(path_in_repo, cache_dir):
    print(f"  downloading {path_in_repo} ...", flush=True)
    return hf_hub_download(REPO, path_in_repo, repo_type="dataset", local_dir=cache_dir)


def get_quarter(conditions, scenes, out, cache, keep_archives):
    target = os.path.join(out, "data_4")
    subset = set(scenes) != set(BLUR_SCENES)
    for c in conditions:
        if all(os.path.isdir(os.path.join(target, c, s)) for s in scenes):
            print(f"[quarter] {c}: already there, skipping")
            continue
        if subset:
            # Only some scenes wanted: stream and stop early instead of fetching the whole archive.
            stream_extract(BASE_URL + f"data_4/data4_{c}.tar.gz", target, scenes, f"[quarter] {c}")
        else:
            print(f"[quarter] {c}")
            arc = fetch(f"data_4/data4_{c}.tar.gz", cache)
            extract_archive(arc, target, keep_scenes=set(scenes))
            if not keep_archives:
                os.remove(arc)


def get_gt(scenes, out, cache, keep_archives):
    for s in scenes:
        target = os.path.join(out, "pointclouds")
        if os.path.isdir(os.path.join(target, s)):
            print(f"[gt] {s}: already there, skipping")
            continue
        print(f"[gt] {s}")
        arc = fetch(f"pointclouds/{s}.tar.gz", cache)
        extract_archive(arc, target)
        if not keep_archives:
            os.remove(arc)


def stream_extract(url, target, scenes, label):
    """Read a .tar.gz while it downloads, keep only the wanted scenes, stop once they have all passed."""
    wanted = set(scenes)
    print(f"{label}: streaming (keeps only {sorted(wanted)}) ...", flush=True)
    done, current = set(), None
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for m in tf:
                parts = m.name.split("/")
                scene = parts[1] if len(parts) >= 2 else None
                if scene != current:
                    if current in wanted:
                        done.add(current)
                        print(f"    finished {current}", flush=True)
                    current = scene
                    if done >= wanted:
                        break
                if scene in wanted:
                    safe_extract(tf, m, target)
    print(f"{label}: done")


def get_baselines(conditions, scenes, out):
    target = os.path.join(out, "baseline_results")
    for c in conditions:
        if all(os.path.isdir(os.path.join(target, c, s)) for s in scenes):
            print(f"[baselines] {c}: already there, skipping")
            continue
        stream_extract(BASE_URL + f"baseline_results/{c}_baseline.tar.gz", target, scenes, f"[baselines] {c}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", nargs="+", default=["quarter", "gt"],
                   choices=["quarter", "gt", "baselines", "all"])
    p.add_argument("--conditions", nargs="+", default=BLUR_CONDITIONS, choices=BLUR_CONDITIONS)
    p.add_argument("--scenes", nargs="+", default=BLUR_SCENES, choices=BLUR_SCENES)
    p.add_argument("--out", default="data", help="output folder (default: ./data)")
    p.add_argument("--keep-archives", action="store_true", help="keep the downloaded .tar.gz files")
    a = p.parse_args()

    what = {"quarter", "gt", "baselines"} if "all" in a.what else set(a.what)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "_archives")
    print(f"Saving into: {os.path.abspath(a.out)}")
    print(f"Scenes: {a.scenes}\nConditions: {a.conditions}\nParts: {sorted(what)}\n")

    if "quarter" in what:
        get_quarter(a.conditions, a.scenes, a.out, cache, a.keep_archives)
    if "gt" in what:
        get_gt(a.scenes, a.out, cache, a.keep_archives)
    if "baselines" in what:
        get_baselines(a.conditions, a.scenes, a.out)
    if not a.keep_archives and os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
