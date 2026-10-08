"""RealX3D scene loading.

Conventions (verified on Ujikintoki, 8 Oct 2026)
------------------------------------------------
* Layout: <data_root>/data_4/<condition>/<Scene>/{train,val,test}/NNNN.JPG, sparse/0 (COLMAP),
  transforms_{train,val,test}.json; ground truth in <data_root>/pointclouds/<Scene>/
  {cull_pointcloud.ply, cull_mesh.ply, colmap2world.npy, depth/NNNN.png}.
* train/ = degraded (blurred) inputs; val/ = SHARP references from the same camera poses (for motion
  blur the blurred content is shifted ~half the blur length: the pose is the END of the exposure)
  (same file names, identical across all 4 blur conditions); test/ = sharp held-out views.
* transforms_*.json poses are in a rescaled/recentred "NeRF" frame (colmap2nerf), NOT the COLMAP
  frame. We therefore read poses from sparse/0 (COLMAP) and map them to the laser frame with
  colmap2world (a similarity: rotation*scale + translation). The result is a METRIC world frame in
  metres. Verified: laser points projected with these cameras match the GT depth maps to ~2 mm.
* GT depth PNGs: uint16 millimetres, full resolution (~4x data_4), z-depth, 0 = invalid.

All cameras returned here are OpenCV-style world-to-camera (x right, y down, z forward), metric.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

CONDITIONS = ["motion_mild", "motion_strong", "defocus_mild", "defocus_strong"]
BLUR_SCENES = ["Ujikintoki", "Laboratory", "Popcorn", "Cupcake",
               "GearWorks", "Chocolate", "MilkCookie", "Limon"]


# ----------------------------------------------------------------------------- cameras
@dataclass
class Camera:
    name: str            # e.g. "0001.JPG"
    K: np.ndarray        # 3x3 intrinsics at data_4 resolution
    W: int
    H: int
    R: np.ndarray        # 3x3 world->camera rotation (metric world frame)
    t: np.ndarray        # 3   world->camera translation (metres)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def viewmat(self) -> np.ndarray:
        """4x4 world-to-camera matrix (what gsplat calls `viewmats`)."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def scaled(self, factor: float) -> "Camera":
        """Camera for an image downscaled by `factor` (e.g. 2 -> half resolution)."""
        K = self.K.copy()
        K[0, :] /= factor
        K[1, :] /= factor
        return Camera(self.name, K, int(round(self.W / factor)), int(round(self.H / factor)),
                      self.R.copy(), self.t.copy())

    def project(self, X: np.ndarray):
        """World points (N,3) -> pixel u, v (N,), depth z (N,) in metres."""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * Xc[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * Xc[:, 1] / z + self.K[1, 2]
        return u, v, z

    def backproject(self, depth: np.ndarray, mask: Optional[np.ndarray] = None, stride: int = 1):
        """Depth map (H,W) in metres -> world points (M,3) and their pixel indices."""
        H, W = depth.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        d = depth[vv, uu]
        m = d > 0
        if mask is not None:
            m &= mask[vv, uu]
        uu, vv, d = uu[m], vv[m], d[m]
        # pixel (i, j) covers [i, i+1) x [j, j+1); its centre is (i+0.5, j+0.5) (COLMAP/gsplat)
        x = (uu + 0.5 - self.K[0, 2]) / self.K[0, 0] * d
        y = (vv + 0.5 - self.K[1, 2]) / self.K[1, 1] * d
        Xc = np.stack([x, y, d], 1)
        Xw = (Xc - self.t) @ self.R          # R^T (Xc - t)
        return Xw, (vv, uu)


def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_text(sparse_dir: str):
    """Return (cameras{id:(model,W,H,params)}, images{name:(R,t,cam_id)}, points (N,3), colors (N,3))."""
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), np.array(list(map(float, p[4:]))))
    imgs = {}
    with open(os.path.join(sparse_dir, "images.txt")) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        imgs[p[9]] = (_qvec2rotmat(q), t, int(p[8]))
    pts, cols = [], []
    pfile = os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(pfile):
        with open(pfile) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                pts.append([float(p[1]), float(p[2]), float(p[3])])
                cols.append([int(p[4]), int(p[5]), int(p[6])])
    return cams, imgs, np.array(pts, dtype=np.float64).reshape(-1, 3), np.array(cols, dtype=np.uint8).reshape(-1, 3)


def read_colmap_binary(sparse_dir: str):
    """Binary fallback (cameras.bin, images.bin, points3D.bin). PINHOLE/SIMPLE_PINHOLE only."""
    def rd(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.bin"), "rb") as f:
        n = rd(f, "Q")[0]
        for _ in range(n):
            cid, model, W, H = rd(f, "iiQQ")
            npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model, 4)
            params = np.array(rd(f, "d" * npar))
            cams[cid] = ({0: "SIMPLE_PINHOLE", 1: "PINHOLE"}.get(model, str(model)), W, H, params)
    imgs = {}
    with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
        n = rd(f, "Q")[0]
        for _ in range(n):
            iid = rd(f, "i")[0]
            q = np.array(rd(f, "dddd"))
            t = np.array(rd(f, "ddd"))
            cid = rd(f, "i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            n2d = rd(f, "Q")[0]
            f.read(24 * n2d)
            imgs[name.decode()] = (_qvec2rotmat(q), t, cid)
    pts, cols = [], []
    pfile = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            n = rd(f, "Q")[0]
            for _ in range(n):
                rd(f, "Q")
                pts.append(rd(f, "ddd"))
                cols.append(rd(f, "BBB"))
                rd(f, "d")
                tl = rd(f, "Q")[0]
                f.read(8 * tl)
    return cams, imgs, np.array(pts, dtype=np.float64).reshape(-1, 3), np.array(cols, dtype=np.uint8).reshape(-1, 3)


def load_colmap2world(*candidates: str) -> np.ndarray:
    for c in candidates:
        if c and os.path.exists(c):
            if c.endswith(".npy"):
                return np.load(c).astype(np.float64)
            return np.loadtxt(c).astype(np.float64)
    raise FileNotFoundError(f"colmap2world not found in {candidates}")


# ----------------------------------------------------------------------------- PLY
def read_ply_points(path: str, with_normals: bool = False):
    """Read x,y,z (and optionally nx,ny,nz) from a binary/ascii PLY (needs `plyfile`)."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    P = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    if with_normals:
        names = v.data.dtype.names
        if "nx" in names:
            N = np.stack([np.asarray(v["nx"]), np.asarray(v["ny"]), np.asarray(v["nz"])], 1)
            return P, N
        return P, None
    return P


# ----------------------------------------------------------------------------- scene
@dataclass
class Scene:
    data_root: str
    scene: str
    condition: str = "motion_strong"
    cams: Dict[str, Camera] = field(default_factory=dict)
    splits: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.dir = os.path.join(self.data_root, "data_4", self.condition, self.scene)
        self.gt_dir = os.path.join(self.data_root, "pointclouds", self.scene)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"Scene folder not found: {self.dir}\n"
                                    f"Download it with scripts/download_realx3d.py")
        sparse = os.path.join(self.dir, "sparse", "0")
        if os.path.exists(os.path.join(sparse, "images.txt")):
            ccams, cimgs, pts, cols = read_colmap_text(sparse)
        else:
            ccams, cimgs, pts, cols = read_colmap_binary(sparse)
        self.c2w_sim = load_colmap2world(os.path.join(self.gt_dir, "colmap2world.npy"),
                                         os.path.join(self.dir, "colmap2world.txt"))
        A, b = self.c2w_sim[:3, :3], self.c2w_sim[:3, 3]
        s = float(np.cbrt(np.linalg.det(A)))
        Q = A / s
        self.colmap_scale = s
        for name, (Rc, tc, cid) in cimgs.items():
            model, W, H, prm = ccams[cid]
            if model == "PINHOLE":
                fx, fy, cx, cy = prm[:4]
            elif model == "SIMPLE_PINHOLE":
                fx = fy = prm[0]
                cx, cy = prm[1:3]
            else:
                raise ValueError(f"Unsupported camera model {model}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            # x_cam_metric = s*(Rc X_col + tc), X_col = Q^T (X_w - b)/s
            R = Rc @ Q.T
            t = -R @ b + s * tc
            self.cams[name] = Camera(name, K, W, H, R, t)
        self.sfm_points = (pts @ A.T + b) if len(pts) else pts
        self.sfm_colors = cols
        for split in ["train", "val", "test"]:
            fn = os.path.join(self.dir, f"transforms_{split}.json")
            if os.path.exists(fn):
                with open(fn) as f:
                    meta = json.load(f)
                self.splits[split] = [os.path.basename(fr["file_path"]) for fr in meta["frames"]]
                if split == "train":
                    self.json_sharpness = {os.path.basename(fr["file_path"]): fr.get("sharpness")
                                           for fr in meta["frames"]}
            else:
                folder = os.path.join(self.dir, split)
                self.splits[split] = sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    # ------------------------------------------------------------------ images
    def image_path(self, name: str, kind: str = "blurred") -> str:
        """kind: 'blurred' (train/ inputs), 'sharp' (val/ references), 'test' (sharp held-out)."""
        sub = {"blurred": "train", "sharp": "val", "test": "test"}[kind]
        return os.path.join(self.dir, sub, name)

    def load_image(self, name: str, kind: str = "blurred", downscale: float = 1.0) -> np.ndarray:
        im = Image.open(self.image_path(name, kind)).convert("RGB")
        if downscale != 1:
            im = im.resize((int(round(im.width / downscale)), int(round(im.height / downscale))),
                           Image.BICUBIC)
        return np.asarray(im, dtype=np.float32) / 255.0

    def camera(self, name: str, downscale: float = 1.0) -> Camera:
        c = self.cams[name]
        return c if downscale == 1 else c.scaled(downscale)

    # ------------------------------------------------------------------ ground truth
    @property
    def has_gt(self) -> bool:
        return os.path.exists(os.path.join(self.gt_dir, "cull_pointcloud.ply"))

    def gt_points(self) -> np.ndarray:
        return read_ply_points(os.path.join(self.gt_dir, "cull_pointcloud.ply"))

    def gt_depth(self, name: str, downscale: float = 1.0) -> Optional[np.ndarray]:
        """GT z-depth (metres) resampled (nearest) to the data_4 image grid (or downscaled). 0 = invalid."""
        stem = os.path.splitext(name)[0]
        p = os.path.join(self.gt_dir, "depth", stem + ".png")
        if not os.path.exists(p):
            return None
        D = np.asarray(Image.open(p)).astype(np.float32) / 1000.0
        cam = self.camera(name, downscale)
        Hf, Wf = D.shape
        v = np.clip(((np.arange(cam.H) + 0.5) * Hf / cam.H).astype(int), 0, Hf - 1)
        u = np.clip(((np.arange(cam.W) + 0.5) * Wf / cam.W).astype(int), 0, Wf - 1)
        return D[v][:, u]


def voxel_downsample(P: np.ndarray, voxel: float, return_index: bool = False):
    """Keep one point per voxel (the first). Fast, no averaging (keeps real samples)."""
    key = np.floor(P / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, idx = np.unique(lin, return_index=True)
    return (P[idx], idx) if return_index else P[idx]
