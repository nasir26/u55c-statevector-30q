"""Host compiler: Qiskit circuit -> engine ops -> passes -> encoded segments.

Program = list of steps executed in order by the runtime:
  ("segment", np.int32 stream)  : one kernel launch (passes over every PC slice)
  ("remap", perm)               : host-mediated qubit permutation; perm[new_phys] = old_phys

Gate categories reported per circuit (R1.1):
  native : 1q gates (U2), controlled-1q (CU2), 2q diagonal (D2), any other 2q unitary (U4)
  relabel: SWAP absorbed into the logical->physical map (no data movement)
  lowered: 3+ qubit gates, decomposed by Qiskit into 1q + CX before compilation
"""
from dataclasses import dataclass, field
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from isa import (MAGIC, HDR_WORDS, PASS_WORDS, OP_WORDS, OP_MUX2, MAX_OPS_PER_PASS, MAX_HIGH)
from scipy.linalg import cossin

SKIP = {"measure", "barrier", "delay", "save_statevector", "id"}
I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
# Qiskit multi-controlled gates whose target is the last qubit and whose controls are all-ones
MC_GATES = {"ccx": lambda g: _X, "c3x": lambda g: _X, "c4x": lambda g: _X, "mcx": lambda g: _X,
            "mcx_gray": lambda g: _X, "mcx_recursive": lambda g: _X, "mcx_vchain": None,
            "ccz": lambda g: _Z, "mcphase": lambda g: np.diag([1, np.exp(1j * float(g.params[0]))]),
            "mcp": lambda g: np.diag([1, np.exp(1j * float(g.params[0]))])}
MC_GATES = {k: v for k, v in MC_GATES.items() if v is not None}


@dataclass
class Op:
    t: int              # target storage qubit
    ctrls: tuple        # control storage qubits (all must be 1 to select M1)
    M0: np.ndarray      # 2x2
    M1: np.ndarray      # 2x2
    src: object = None  # original qiskit instruction (for lowering fallback)

    @property
    def q(self):
        return (self.t,) + tuple(self.ctrls)


@dataclass
class Stats:
    n_input_gates: int = 0
    native: dict = field(default_factory=dict)
    relabel: int = 0
    lowered: dict = field(default_factory=dict)
    csd_2q: int = 0
    fused_1q: int = 0
    n_ops: int = 0
    n_passes: int = 0
    n_remaps: int = 0
    n_segments: int = 0


def _bump(d, k):
    d[k] = d.get(k, 0) + 1


def controlled_target(M, nq):
    """If M (Qiskit order, 2^nq) is a multi-controlled 2x2 on one qubit, return (j, U): target j,
    controls = all others. Else None."""
    full = (1 << nq) - 1
    for j in range(nq):
        on = full & ~(1 << j)          # all other bits 1
        i0, i1 = on, on | (1 << j)
        R = M.copy()
        U = M[np.ix_([i0, i1], [i0, i1])].copy()
        R[np.ix_([i0, i1], [i0, i1])] = I2
        if np.allclose(R, np.eye(1 << nq), atol=1e-12):
            return j, U
    return None


def ops_2q(M, a, b, stats):
    """M: 4x4, basis bit_a + 2*bit_b. Returns list of Op (1 op if diagonal/controlled, else 3 via CSD)."""
    if np.allclose(M, np.diag(np.diag(M)), atol=1e-12):
        d = np.diag(M)
        return [Op(b, (a,), np.diag([d[0], d[2]]), np.diag([d[1], d[3]]))]
    ct = controlled_target(M, 2)
    if ct is not None:
        j, U = ct
        t, c = (a, b) if j == 0 else (b, a)
        return [Op(t, (c,), I2, U)]
    (u1, u2), th, (v1h, v2h) = cossin(M, p=2, q=2, separate=True)
    c, s_ = np.cos(th), np.sin(th)
    R0 = np.array([[c[0], -s_[0]], [s_[0], c[0]]]); R1 = np.array([[c[1], -s_[1]], [s_[1], c[1]]])
    stats.csd_2q += 1
    return [Op(a, (b,), v1h, v2h), Op(b, (a,), R0, R1), Op(a, (b,), u1, u2)]


def circuit_to_ops(qc: QuantumCircuit, stats: Stats, fuse_1q=True, max_width=None):
    """Returns (ops, relab); SWAPs are absorbed into a relabelling: relab[l] = storage qubit of logical l."""
    n = qc.num_qubits
    relab = list(range(n))
    ops = []
    pending = {}

    def flush(q):
        if q in pending:
            U = pending.pop(q)
            ops.append(Op(q, (), U, U))

    def emit(op_list):
        for op in op_list:
            for q in op.q:
                flush(q)
            ops.append(op)

    def handle(inst, qs, lowered_from=None):
        name = inst.operation.name
        if name in SKIP:
            return
        if lowered_from is None:
            stats.n_input_gates += 1
        if name == "swap":
            a, b = qs
            relab[a], relab[b] = relab[b], relab[a]
            stats.relabel += 1
            return
        st = [relab[q] for q in qs]
        # multi-controlled X / Z / phase: built directly as MUX2 (no dense operator), native if it fits the window
        if len(qs) > 2 and name in MC_GATES and (max_width is None or len(qs) <= max_width):
            if lowered_from is None:
                _bump(stats.native, name)
            U = MC_GATES[name](inst.operation)
            emit([Op(st[-1], tuple(st[:-1]), I2, U, src=(inst, qs))])
            return
        if len(qs) > 2 and ((max_width is not None and len(qs) > max_width) or len(qs) > 10):
            if lowered_from is None:
                _bump(stats.native, name)
            lower(inst, qs)            # wider than the tile window: never build the dense matrix
            return
        M = Operator(inst.operation).data
        if lowered_from is None:
            _bump(stats.native, name)
        if len(qs) == 1:
            s = st[0]
            if fuse_1q and s in pending:
                pending[s] = M @ pending[s]; stats.fused_1q += 1
            elif fuse_1q:
                pending[s] = M
            else:
                ops.append(Op(s, (), M, M))
            return
        if len(qs) == 2:
            emit(ops_2q(M, st[0], st[1], stats))
            return
        ct = controlled_target(M, len(qs))
        if ct is not None:
            j, U = ct
            emit([Op(st[j], tuple(x for i, x in enumerate(st) if i != j), I2, U, src=(inst, qs))])
            return
        lower(inst, qs)

    def lower(inst, qs):
        stats.native[inst.operation.name] -= 1
        _bump(stats.lowered, inst.operation.name)
        sub = QuantumCircuit(len(qs))
        sub.append(inst.operation, list(range(len(qs))))
        sub = transpile(sub, basis_gates=["u", "cx"], optimization_level=1)
        for si in sub.data:
            handle(si, [qs[sub.find_bit(q).index] for q in si.qubits], lowered_from=inst)

    for inst in qc.data:
        handle(inst, [qc.find_bit(q).index for q in inst.qubits])
    for q in list(pending):
        flush(q)
    stats.n_ops = len(ops)
    return ops, relab


def _pass_m(phys_set, k, M=None):
    """Smallest m in {0..M} with |{q >= k-m}| <= m, plus the high qubits; None if infeasible."""
    M = MAX_HIGH if M is None else M
    for m in range(M + 1):
        hi = sorted(q for q in phys_set if q >= k - m)
        if len(hi) <= m:
            return m, hi
    return None


def schedule(ops, n, L, k, stats: Stats, remap_policy="lru", tiling=True, l2p0=None, M=MAX_HIGH):
    """Map logical ops to physical passes. Physical qubits >= L are bank qubits (remap needed).

    p2l / l2p: physical <-> storage-logical maps. Returns steps list."""
    l2p = list(l2p0) if l2p0 is not None else list(range(n))
    steps = []
    cur_passes = []     # passes of the current segment: list of (ops_phys)
    cur = []            # ops in the open pass (physical qubits)
    cur_set = set()
    k = min(k, L)

    def close_pass():
        nonlocal cur, cur_set
        if cur:
            cur_passes.append(cur)
        cur, cur_set = [], set()

    def close_segment():
        close_pass()
        if cur_passes:
            steps.append(("segment", list(cur_passes)))
            cur_passes.clear()

    def dist(p, idx):
        l = l2p.index(p)
        for j in range(idx, len(ops)):
            if l in ops[j].q:
                return j - idx
        return 10 ** 9

    for idx, op in enumerate(ops):
        phys = [l2p[q] for q in op.q]
        if any(p >= L for p in phys) or _pass_m(set(phys), k, M) is None:
            low = k - M
            assert len(phys) <= low + M, "op wider than the tile window"
            far = lambda p: -dist(p, idx)
            bank = [p for p in phys if p >= L]
            high_local = [p for p in phys if low <= p < L]
            hi_slots = sorted((p for p in range(max(k, L - 6), L) if p not in phys), key=far)
            if bank and len(high_local) + len(bank) <= M and len(hi_slots) >= len(bank):
                # fast path: each bank qubit swaps with a high local qubit (large contiguous chunks)
                move, vic = bank, hi_slots[:len(bank)]
            else:
                # general path: keep at most two local qubits at/above the low window; move the rest
                # (incl. bank qubits) into the low window, evicting the qubits used furthest ahead
                keep = high_local[:M]
                move = [p for p in phys if p >= low and p not in keep]
                low_slots = sorted((p for p in range(low) if p not in phys), key=far)
                high_slots = sorted((p for p in range(low, L) if p not in phys), key=far)
                high_free = M - len(keep)
                vic = []
                for _ in move:
                    if low_slots:
                        vic.append(low_slots.pop(0))
                    elif high_free > 0 and high_slots:
                        vic.append(high_slots.pop(0)); high_free -= 1
                    else:
                        raise RuntimeError("cannot fit op into tile window")
            close_segment()
            perm = list(range(n))           # perm[new_phys] = old_phys
            for mq, vq in zip(move, vic):
                perm[mq], perm[vq] = vq, mq
            steps.append(("remap", perm))
            stats.n_remaps += 1
            inv = {old: new for new, old in enumerate(perm)}
            l2p = [inv[p] for p in l2p]
            phys = [l2p[q] for q in op.q]
        new_set = cur_set | set(phys)
        if (not tiling and cur) or len(cur) >= MAX_OPS_PER_PASS or _pass_m(new_set, k, M) is None:
            close_pass()
            new_set = set(phys)
        cur.append((op, phys))
        cur_set = new_set
    close_segment()
    return steps, l2p


def encode_segment(passes, L, k, M=MAX_HIGH):
    words = [MAGIC, L, len(passes), k, 0, 0, 0, 0]
    for p in passes:
        phys_set = set(q for _, ph in p for q in ph)
        m, hi = _pass_m(phys_set, k, M)
        h = hi + [0] * (4 - len(hi))
        tmap = {q: q for q in phys_set if q < k - m}
        for i, q in enumerate(hi):
            tmap[q] = k - m + i
        words += [m, h[0], h[1], len(p), h[2], h[3], 0, 0]
        for op, ph in p:
            tq = [tmap[q] for q in ph]
            cmask = 0
            for c in tq[1:]:
                cmask |= 1 << c
            fl = np.zeros(16, dtype=np.float32)
            for i, Mat in enumerate((op.M0, op.M1)):
                f = np.asarray(Mat, dtype=np.complex64).reshape(-1)
                fl[8 * i:8 * i + 8:2] = f.real
                fl[8 * i + 1:8 * i + 8:2] = f.imag
            words += [OP_MUX2, tq[0], cmask, 0, 0, 0, 0, 0] + list(fl.view(np.int32))
    arr = np.asarray(words, dtype=np.int64)
    return (arr & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


def compile_circuit(qc, L, k, fuse_1q=True, tiling=True, remap_policy="lru", initial_map=None, max_high=MAX_HIGH):
    """Full pipeline. initial_map: list l2p for the initial logical->physical placement (layout ablation)."""
    k = min(k, L)
    # high-qubit slots: bounded by the kernel build (max_high) and by the sub-block size (k - m - 2 >= 1 words)
    max_high = max(0, min(max_high, k - 3))
    stats = Stats()
    # an op must fit the tile window: at most k-2 low qubits + 2 high qubits
    ops, relab = circuit_to_ops(qc, stats, fuse_1q, max_width=k)
    n = qc.num_qubits
    steps, l2p_end = schedule(ops, n, L, k, stats, remap_policy, tiling, initial_map, max_high)
    prog = []
    if initial_map is not None and list(initial_map) != list(range(n)):
        # host places logical qubit l at physical initial_map[l] before the first launch
        perm = [0] * n
        for l, p in enumerate(initial_map):
            perm[p] = l
        prog.append(("remap", perm))
    for kind, payload in steps:
        if kind == "segment":
            prog.append(("segment", encode_segment(payload, L, k, max_high)))
            stats.n_segments += 1
            stats.n_passes += len(payload)
        else:
            prog.append(("remap", payload))
    # final logical qubit l lives at physical position l2p_end[relab[l]]
    final_map = [l2p_end[relab[l]] for l in range(n)]
    return prog, final_map, stats
