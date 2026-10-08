"""Numpy golden model of the SV kernel: executes an encoded segment exactly as an engine does,
tile by tile, in complex64 arithmetic. Used to validate the compiler (vs Qiskit) and the HLS C-sim."""
import numpy as np
from isa import MAGIC, HDR_WORDS, PASS_WORDS, OP_WORDS, OP_MUX2


def decode(seg):
    seg = np.asarray(seg, dtype=np.int32)
    assert seg[0] == MAGIC
    L, npass, k = int(seg[1]), int(seg[2]), int(seg[3])
    pos = HDR_WORDS
    passes = []
    for _ in range(npass):
        m, h0, h1, nops, h2, h3 = (int(x) for x in seg[pos:pos + 6]); pos += PASS_WORDS
        ops = []
        for _ in range(nops):
            rec = seg[pos:pos + OP_WORDS]; pos += OP_WORDS
            f = rec[8:24].view(np.float32)
            c = (f[0::2] + 1j * f[1::2]).astype(np.complex64)
            ops.append((int(rec[0]), int(rec[1]), int(rec[2]), c[:4].reshape(2, 2), c[4:].reshape(2, 2)))
        passes.append((m, [h0, h1, h2, h3][:m], ops))
    return L, k, passes


def tile_index_map(L, k, m, hi):
    """Returns idx[t, j]: local slice index of tile-element j in tile t (shape 2^(L-k) x 2^k)."""
    hi = list(hi)[:m]
    low = k - m
    used = set(range(low)) | set(hi)
    free = [b for b in range(L) if b not in used]
    assert len(free) == L - k
    j = np.arange(1 << k, dtype=np.int64)
    off = j & ((1 << low) - 1)
    for i, h in enumerate(hi):
        off |= ((j >> (low + i)) & 1) << h
    t = np.arange(1 << (L - k), dtype=np.int64)
    base = np.zeros_like(t)
    for i, b in enumerate(free):
        base |= ((t >> i) & 1) << b
    return base[:, None] | off[None, :]


def apply_op_tile(T, code, t, cmask, M0, M1):
    """MUX2 on tile qubit t: pairs whose index has all cmask bits set use M1, others M0."""
    assert code == OP_MUX2
    nt, size = T.shape
    k = size.bit_length() - 1
    V = T.reshape(nt, 1 << (k - t - 1), 2, 1 << t)
    i0 = np.arange(size).reshape(1 << (k - t - 1), 2, 1 << t)[:, 0, :]
    sel = ((i0 & cmask) == cmask) if cmask else np.zeros_like(i0, dtype=bool)
    x0, x1 = V[:, :, 0, :].copy(), V[:, :, 1, :].copy()
    for M, mask in ((M0, ~sel), (M1, sel)):
        y0 = M[0, 0] * x0 + M[0, 1] * x1
        y1 = M[1, 0] * x0 + M[1, 1] * x1
        V[:, :, 0, :] = np.where(mask[None], y0, V[:, :, 0, :])
        V[:, :, 1, :] = np.where(mask[None], y1, V[:, :, 1, :])


def run_segment(state, P, seg):
    """state: full complex64 vector, length P * 2^L. Each PC slice processed independently."""
    L, k, passes = decode(seg)
    assert state.size == P << L
    for m, hi, ops in passes:
        idx = tile_index_map(L, k, m, hi)
        for p in range(P):
            sl = state[p << L:(p + 1) << L]
            T = sl[idx]
            for code, t, cm, M0, M1 in ops:
                apply_op_tile(T, code, t, cm, M0, M1)
            sl[idx] = T
    return state


def remap(state, n, perm):
    """perm[new_phys] = old_phys: new bit position new_phys takes the old bit old_phys."""
    v = state.reshape([2] * n)          # axis i corresponds to bit n-1-i
    axes = [n - 1 - perm[n - 1 - i] for i in range(n)]
    return np.ascontiguousarray(np.transpose(v, axes)).reshape(-1)


def unpermute(state, n, final_map):
    """final_map[l] = physical position of logical qubit l -> return state in logical order."""
    perm = [final_map[l] for l in range(n)]   # new_phys = l takes old bit final_map[l]
    return remap(state, n, perm)


def run_program(prog, n, P, final_map, state=None):
    if state is None:
        state = np.zeros(1 << n, dtype=np.complex64); state[0] = 1
    for kind, payload in prog:
        if kind == "segment":
            run_segment(state, P, payload)
        else:
            state = remap(state, n, payload)
    return unpermute(state, n, final_map)
