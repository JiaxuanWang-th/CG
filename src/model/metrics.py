"""Training metrics aligned with official StraightPCF (unit-sphere Chamfer)."""
import numpy as np
from scipy.spatial import cKDTree

from ..data.utils import normalize_unit_sphere


def chamfer_distance_unit_sphere(gen, ref):
    """
    Bidirectional Chamfer on unit sphere (same normalization as official SPCF / evaluate.py).

    Args:
        gen: (N, 3) denoised
        ref: (M, 3) clean
    Returns:
        scalar CD
    """
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ref_n, center, scale = normalize_unit_sphere(ref)
    if scale < 1e-12:
        return 0.0
    gen_n = (gen - center) / scale

    tree_b = cKDTree(ref_n)
    dist_a2b, _ = tree_b.query(gen_n, k=1)
    tree_a = cKDTree(gen_n)
    dist_b2a, _ = tree_a.query(ref_n, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())
