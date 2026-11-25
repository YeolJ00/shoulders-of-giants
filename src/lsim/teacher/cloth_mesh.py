"""The cloth grid as a mesh, numpy only (no warp: CPU tools such as the real-data converters import it without
touching a GPU). lsim.teacher.cloth_teacher re-exports both."""
import numpy as np


def grid(nx, ny, h):
    """A flat nx x ny cell grid in its own x-y plane, diagonals alternating (no preferred shear): vertices
    (V, 3) row by row along x, triangles (T, 3) counter-clockwise seen from +z."""
    v = np.array([(i * h, j * h, 0.0) for j in range(ny + 1) for i in range(nx + 1)], np.float64)
    t = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i, (j + 1) * (nx + 1) + i + 1
            t += [(a, b, d), (a, d, c)] if (i + j) % 2 == 0 else [(a, b, c), (b, d, c)]
    return v, np.array(t, np.int32)


def mesh_edges(tris):
    """Unique edges (E, 2) of a triangle mesh and each triangle's three edge indices (T, 3)."""
    tris = np.asarray(tris, np.int64).reshape(-1, 3)
    pairs = np.sort(np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    edges, inv = np.unique(pairs, axis=0, return_inverse=True)
    return edges.astype(np.int32), inv.reshape(-1, 3).astype(np.int32)
