"""V0 unit tests for the strata (run: python -m pytest tests -q)."""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp import strata as S  # noqa: E402


def plate(z, half=0.3, step=0.003):
    g = np.arange(-half, half, step)
    X, Y = np.meshgrid(g, g)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)


def _centre_idx(P, z):
    return np.argmin(np.linalg.norm(P - np.array([0, 0, z]), axis=1))


def test_plane_curvature_near_zero():
    P = plate(0.0)
    tree = cKDTree(P)
    N, sig = S.pca_normals_curvature(P[:100], tree, P, k=48)
    assert np.all(sig < 1e-6)
    assert np.allclose(np.abs(N[:, 2]), 1, atol=1e-6)


def test_two_plates_clearance_and_open_back():
    # surface at z=0 facing +z (camera above); another plate 4 cm above it; nothing below
    P = np.concatenate([plate(0.0), plate(0.04)])
    occ = S.Occupancy(P, voxel=0.01)
    i = _centre_idx(P, 0.0)
    Q, N = P[i:i + 1], np.array([[0, 0, 1.0]])
    front, back = S.proximity(occ, Q, N)
    assert abs(front[0] - 0.04) <= 0.012, front
    assert np.isinf(back[0]), back


def test_slab_thickness():
    # 4 cm thick slab: top at z=0 (normal +z), bottom at z=-0.04; open above
    P = np.concatenate([plate(0.0), plate(-0.04)])
    occ = S.Occupancy(P, voxel=0.01)
    i = _centre_idx(P, 0.0)
    front, back = S.proximity(occ, P[i:i + 1], np.array([[0, 0, 1.0]]))
    assert np.isinf(front[0]), front
    assert abs(back[0] - 0.04) <= 0.012, back


def test_front_and_back_differ():
    P = np.concatenate([plate(0.0), plate(0.06), plate(-0.15)])
    occ = S.Occupancy(P, voxel=0.01)
    i = _centre_idx(P, 0.0)
    front, back = S.proximity(occ, P[i:i + 1], np.array([[0, 0, 1.0]]))
    assert abs(front[0] - 0.06) <= 0.012 and abs(back[0] - 0.15) <= 0.012, (front, back)
