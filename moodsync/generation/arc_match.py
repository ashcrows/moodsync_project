"""Arc Match Score — the project's novel evaluation metric.

    ArcMatch(T, G) = cos_sim( flatten(T), flatten(G) )

Both the target arc T and the generated (re-analysed) arc G are resampled to the
same number of points, flattened to a (2*N,) vector, and compared with cosine
similarity. Score in [-1, 1]; 1.0 = perfect emotional match.
"""
from __future__ import annotations

import numpy as np


def resample_arc(arc: np.ndarray, n_points: int) -> np.ndarray:
    """Resample an (T, 2) arc to (n_points, 2) by linear interpolation."""
    arc = np.asarray(arc, dtype=np.float32).reshape(-1, 2)
    if len(arc) == n_points:
        return arc
    xs = np.linspace(0, 1, len(arc))
    q = np.linspace(0, 1, n_points)
    v = np.interp(q, xs, arc[:, 0])
    a = np.interp(q, xs, arc[:, 1])
    return np.stack([v, a], axis=1).astype(np.float32)


def arc_match_score(target: np.ndarray, generated: np.ndarray, n_points: int = 32) -> float:
    """Cosine similarity between target and generated emotion trajectories.

    NOTE: this is *uncentered* cosine, so it is dominated by the shared mean
    (DC) level of the two arcs rather than by their shape. A completely flat
    generated arc carrying no trajectory information still scores ~+0.42
    against a typical target. Report it alongside `arc_shape_score`, which
    removes that offset — see that function for the rationale.
    """
    t = resample_arc(target, n_points).reshape(-1)
    g = resample_arc(generated, n_points).reshape(-1)
    denom = (np.linalg.norm(t) * np.linalg.norm(g)) + 1e-8
    return float(np.dot(t, g) / denom)


def arc_shape_score(target: np.ndarray, generated: np.ndarray, n_points: int = 32) -> float:
    """Trajectory-only match: cosine after removing each arc's per-dimension mean.

    `arc_match_score` answers "is the overall emotional level similar?"; this
    answers "does the emotion *move* the way it was asked to?" — which is the
    claim the Arc Match Score is meant to support. Centering makes a flat
    (trajectory-free) arc score ~0.0 instead of ~+0.42, and is equivalent to a
    Pearson correlation over the flattened arc.

    Returns 0.0 when either arc is constant, since a constant has no shape to
    compare rather than a perfectly matching one.
    """
    t = resample_arc(target, n_points).astype(np.float64)
    g = resample_arc(generated, n_points).astype(np.float64)
    t = (t - t.mean(axis=0)).reshape(-1)
    g = (g - g.mean(axis=0)).reshape(-1)
    nt, ng = np.linalg.norm(t), np.linalg.norm(g)
    if nt < 1e-8 or ng < 1e-8:
        return 0.0
    return float(np.dot(t, g) / (nt * ng))
