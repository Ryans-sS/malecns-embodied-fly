"""Compiled inner loop for synaptic propagation.

Runtime is dominated by scattering each spiking cell's outgoing weights into the
membrane-potential vector - roughly 127,000 random-access additions per simulated
millisecond. NumPy cannot express that without materialising a large index array
and running bincount over it; a compiled loop does it in place.

Falls back to the NumPy path if numba is unavailable, so numba stays optional.
"""
import numpy as np

try:
    from numba import njit

    @njit(cache=True, fastmath=True, nogil=True)
    def _propagate(spiking, out_start, out_end, edge_post, edge_w, v):
        for k in range(spiking.shape[0]):
            i = spiking[k]
            for e in range(out_start[i], out_end[i]):
                v[edge_post[e]] += edge_w[e]

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    HAVE_NUMBA = False
    _propagate = None


def propagate(spiking, out_start, out_end, edge_post, edge_w, v, n):
    """Add every spiking cell's outgoing weights into v, in place."""
    if HAVE_NUMBA:
        _propagate(spiking, out_start, out_end, edge_post, edge_w, v)
        return
    s0 = out_start[spiking]
    e0 = out_end[spiking]
    ln = e0 - s0
    if not ln.all():
        nz = ln > 0
        s0, e0, ln = s0[nz], e0[nz], ln[nz]
    tot = int(ln.sum())
    if not tot:
        return
    g = np.ones(tot, dtype=np.int64)
    g[0] = s0[0]
    if len(s0) > 1:
        g[np.cumsum(ln)[:-1]] = s0[1:] - e0[:-1] + 1
    np.cumsum(g, out=g)
    v += np.bincount(edge_post[g], weights=edge_w[g], minlength=n).astype(np.float32)
