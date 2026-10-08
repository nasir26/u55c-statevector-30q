"""Host runtime for the SV kernels (sv_engine CUs + optional sv_swap CUs) on one U55C.

Timing fields (revision Sec. 6), all per circuit, excluding T_setup:
  T_compile : host compile (gate -> MUX2 ops, fusion, pass scheduling, encoding)
  T_h2d     : initial-state upload + instruction uploads
  T_kernel  : sum over segments of (start all active engine CUs -> all finished)
  T_exchange: bank-qubit remaps (device sv_swap runs, or host-mediated transfers)
  T_d2h     : final statevector read-back
  T_post    : Born-rule sampling of 1024 shots from the physical-order slices (hierarchical), mapped to logical
              bitstrings. Re-ordering the full state into logical order (only for fidelity checks) is reported
              separately as T_state_reorder and excluded from T_e2e.
  T_h2d     : device-side zeroing pass + 32-byte write of amplitude 0 + instruction uploads
"""
import uuid  # noqa: F401  must precede pyxrt device init (docs/BASELINE_AUDIT.md A)
import os, sys, time
import numpy as np
import pyxrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiler import compile_circuit  # noqa: E402
from isa import MAGIC, OP_MUX2  # noqa: E402
from model import remap as host_remap  # noqa: E402

TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
KMAX = 12
LPREF = 17        # preferred minimum slice size (local qubits): below it, extra slices cost more in remaps than they
                  # save in kernel time (measured P-sweep, results/raw/explore/planner_psweep_m4.json)
LMIN = 12         # minimum local qubits per PC slice: n <= 12 runs on one PC (padded), so the tile window
                  # (k = 12: 10 low + 2 high qubits) holds wide multi-controlled gates natively
AMP = 8           # bytes per complex64 amplitude
WORD_SWAP = 64    # sv_swap word size (bytes)


def transpositions(perm):
    """perm[new] = old  ->  list of (a, b) bit transpositions whose product is perm."""
    p = list(perm)
    out = []
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            out.append((min(i, j), max(i, j)))
            p[i], p[j] = p[j], p[i]
    return out


def sample_slices(slices, shots, fmap, n_run, n, seed=None, chunk_bits=12):
    """Born-rule sampling of `shots` outcomes from a statevector stored as physical-order slices.
    Hierarchical: float64 sums per chunk of 2^chunk_bits amplitudes, choose chunks, then positions inside the
    chosen chunks. Sampled physical indices are mapped to logical bitstrings (fmap[l] = physical position of
    logical qubit l); padded qubits (l >= n) are dropped. Returns {logical index: count}."""
    if shots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    L = slices[0].size.bit_length() - 1
    cb = min(chunk_bits, L)
    from concurrent.futures import ThreadPoolExecutor

    def chunk_sums(s):   # |a|^2 = re^2 + im^2 on the float32 view (no sqrt); float32 within a chunk, float64 across
        f = s.view(np.float32).reshape(-1, 2 << cb)
        return np.einsum("ij,ij->i", f, f).astype(np.float64)
    with ThreadPoolExecutor(max_workers=min(16, len(slices))) as ex:
        sums = np.concatenate(list(ex.map(chunk_sums, slices)))
    cs = np.cumsum(sums)
    u = rng.random(shots) * cs[-1]
    ch = np.searchsorted(cs, u, side="right").clip(0, sums.size - 1)
    rem = u - np.where(ch > 0, cs[ch - 1], 0.0)
    phys = np.empty(shots, dtype=np.int64)
    per_slice = (1 << L) >> cb
    for c in np.unique(ch):
        sel = np.nonzero(ch == c)[0]
        s = slices[c // per_slice]
        off = (c % per_slice) << cb
        p = np.cumsum(np.abs(s[off:off + (1 << cb)].astype(np.complex128)) ** 2)
        phys[sel] = (c << cb) + np.searchsorted(p, rem[sel], side="right").clip(0, (1 << cb) - 1)
    logical = np.zeros(shots, dtype=np.int64)
    for l in range(n):
        logical |= ((phys >> fmap[l]) & 1) << l
    vals, cnt = np.unique(logical, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


class SVRuntime:
    def __init__(self, bdf, xclbin, n_engines, n_swap=0, use_device_swap=True, max_high=None):
        # max high qubits per pass must match the bitstream's MAXH: read from <build>/maxh.txt (absent -> 2)
        mf = os.path.join(os.path.dirname(os.path.abspath(xclbin)), "maxh.txt")
        build_maxh = int(open(mf).read()) if os.path.exists(mf) else 2
        if max_high is not None and max_high > build_maxh:
            raise ValueError(f"max_high={max_high} exceeds the build's MAXH={build_maxh}")
        self.max_high = build_maxh if max_high is None else max_high
        if bdf[5:7] in ("01", "21"):
            raise RuntimeError("cards 01/21 are not allowed (author decision)")
        t0 = time.perf_counter()
        self.dev = pyxrt.device(bdf)
        self.uuid = self.dev.load_xclbin(xclbin)
        self.eng = [pyxrt.kernel(self.dev, self.uuid, f"sv_engine:{{sv_engine_{i + 1}}}") for i in range(n_engines)]
        self.swp = [pyxrt.kernel(self.dev, self.uuid, f"sv_swap:{{sv_swap_{i + 1}}}") for i in range(n_swap)]
        self.use_device_swap = use_device_swap and n_swap > 0
        self.st, self.ins = [], []
        self.L = None; self.P = None
        self.pcs = self._read_pcs(xclbin, n_engines)     # [(first_pc, n_pcs)] per engine CU, from the build's link.cfg
        self.swap_ok = None                              # result of the device-swap self-test for the current buffers
        self.log = []
        self.T_setup = time.perf_counter() - t0

    @staticmethod
    def _read_pcs(xclbin, n):
        import re
        cfg = os.path.join(os.path.dirname(os.path.abspath(xclbin)), "link.cfg")
        pcs = [(None, 1)] * n
        if os.path.exists(cfg):
            for line in open(cfg):
                m = re.match(r"sp=sv_engine_(\d+)\.st:HBM\[(\d+)(?::(\d+))?\]", line.strip())
                if m and int(m.group(1)) <= n:
                    a = int(m.group(2)); b = int(m.group(3)) if m.group(3) else a
                    pcs[int(m.group(1)) - 1] = (a, b - a + 1)
        return pcs

    def _slot(self, s):
        """slice s -> (engine index, explicit PC bank or None). Slices are dealt round-robin over engines first,
        so P <= n_engines uses one slice per engine (its first PC)."""
        ne = len(self.eng)
        e, j = s % ne, s // ne
        first, npcs = self.pcs[e]
        return e, (None if first is None else first + j)

    def _bank_for(self, i, nbytes):
        """Explicit PC bank when the buffer fits in one 512 MB PC; else the CU's (multi-PC) memory group.
        Buffers in a multi-PC group are NOT accessible to the sv_swap CUs (XRT silently makes a local copy), so
        device swaps are disabled for them (see docs/BASELINE_AUDIT.md H)."""
        first, npcs = self.pcs[i]
        if first is not None and nbytes <= (512 << 20):
            return first, True
        return self.eng[i].group_id(0), npcs == 1

    def _swap_selftest(self):
        """Run one small device swap between the first two engine buffers and verify it (guards against XRT
        allocating a private copy for the swap kernel)."""
        if self.P < 2 or not self.swp:
            return False
        a, b = self.st[0], self.st[1]
        va = np.frombuffer(a.map(), dtype=np.uint8, count=128); vb = np.frombuffer(b.map(), dtype=np.uint8, count=128)
        va[:] = 0xA5; vb[:] = 0x5A
        a.sync(TO, 128, 0); b.sync(TO, 128, 0)
        self.swp[0](a, b, 0, 0, 2).wait()
        a.sync(FROM, 128, 0); b.sync(FROM, 128, 0)
        ok = bool((va == 0x5A).all() and (vb == 0xA5).all())
        va[:] = 0; vb[:] = 0
        a.sync(TO, 128, 0); b.sync(TO, 128, 0)
        return ok

    # ------------------------------------------------------------------ allocation
    @property
    def n_slots(self):
        """Slices the build can hold in single-PC buffers: engines x PCs per engine (virtual slices)."""
        return sum(npcs for _, npcs in self.pcs)

    def plan(self, n, p_max=None):
        """P slices: one per engine when the slice fits a single PC; virtual slices (P > engines) only when the slice
        would not fit (measured: fewer bank qubits -> fewer remaps; P=16 beats P=32 at n=28 by 6-9% e2e)."""
        cap = p_max or self.n_slots
        P = 1
        while P * 2 <= min(cap, len(self.eng)) and n - (P * 2).bit_length() + 1 >= LPREF:
            P *= 2
        def fits(P):
            L = max(n, LMIN + P.bit_length() - 1) - (P.bit_length() - 1)
            return (1 << L) * AMP + (4 << 20) <= (512 << 20)
        while not fits(P) and P * 2 <= cap and n - (P * 2).bit_length() + 1 >= LMIN:
            P *= 2
        n_run = max(n, LMIN + P.bit_length() - 1)
        return P, n_run, n_run - (P.bit_length() - 1)

    def _alloc(self, P, L):
        if self.P == P and self.L == L:
            return
        # release the previous BOs first: an old and a new slice of the same CU must not coexist in one 512 MB PC
        self.st = []
        import gc; gc.collect()
        # one BO per engine: state slice followed by the instruction region (single AXI port per CU)
        self.ins_cap = 4 << 20
        nbytes = (1 << L) * AMP + self.ins_cap
        if P > len(self.eng):
            # virtual slices: each engine owns several single-PC buffers, processed one after the other
            assert nbytes <= (512 << 20), "virtual slice larger than one PC"
            self.slot = [self._slot(s) for s in range(P)]
            banks = [(b, True) for _, b in self.slot]
        else:
            self.slot = [(s, None) for s in range(P)]
            banks = [self._bank_for(i, nbytes) for i in range(P)]
        self.st = [pyxrt.bo(self.dev, nbytes, pyxrt.bo.flags.normal, bank) for bank, _ in banks]
        swappable = all(ok for _, ok in banks)
        self.P, self.L = P, L
        self.order = list(range(P))   # order[logical PC index] = physical BO index
        for i in range(P):            # one-time host zero upload, so device memory never holds NaN garbage
            self._view(i)[:] = 0
            self.st[i].sync(TO, (1 << L) * AMP, 0)
        self.swap_ok = bool(self.swp) and swappable and self._swap_selftest()
        if self.swp and not self.swap_ok:
            self.log.append(f"device swap unavailable for P={P}, L={L} (multi-PC buffers or self-test failed): host exchange")

    def _view(self, i):
        return np.frombuffer(self.st[i].map(), dtype=np.complex64, count=1 << self.L)

    # ------------------------------------------------------------------ steps
    def _zero_segment(self):
        """One pass with MUX2(0, 0, Z, Z), Z = 0: overwrites every amplitude with 0 on the device."""
        L = self.L; k = min(KMAX, L)
        w = [MAGIC, L, 1, k, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [OP_MUX2, 0, 0, 0, 0, 0, 0, 0] + [0] * 16
        return np.asarray(w, dtype=np.int64).astype(np.int32)

    def _init_zero_state(self):
        """|0...0> without a full host upload: device zeroing pass + one 32-byte write of amplitude 0."""
        self._upload_ins(self._zero_segment())
        self._run_segment()
        v = self._view(0)
        v[:4] = 0; v[0] = 1
        self.st[0].sync(TO, 32, 0)
    def _upload_ins(self, seg):
        nb = seg.nbytes
        if nb > self.ins_cap:
            raise RuntimeError(f"segment of {nb} B exceeds instruction region {self.ins_cap} B")
        off = (1 << self.L) * AMP
        for i in range(self.P):
            np.frombuffer(self.st[i].map(), dtype=np.int32, count=seg.size, offset=off)[:] = seg
            self.st[i].sync(TO, nb, off)

    def _run_segment(self):
        # every slice gets the same segment; engine i owns BO i (its PC), whatever its logical order
        ins_off = (1 << self.L) * AMP // 32      # 256-bit word offset of the instruction region
        # slice s runs on its engine; several slices of one engine are queued on that CU and execute in order
        runs = [self.eng[self.slot[i][0]](self.st[i], ins_off) for i in range(self.P)]
        for r in runs:
            r.wait()

    def _swap_bank_local(self, j, v):
        """Transposition of bank bit j (PC index bit) with local bit v."""
        L = self.L
        cw = (1 << v) * AMP           # chunk bytes
        nchunk = 1 << (L - v - 1)
        pairs = [(p, p | (1 << j)) for p in range(self.P) if not (p >> j) & 1]
        if self.use_device_swap and self.swap_ok and cw % WORD_SWAP == 0:
            runs = []
            q = 0
            for p, pp in pairs:
                a, b = self.st[self.order[p]], self.st[self.order[pp]]
                for c in range(nchunk):
                    offa = ((2 * c + 1) * cw) // WORD_SWAP   # chunk with bit v = 1 in PC p
                    offb = (2 * c * cw) // WORD_SWAP         # chunk with bit v = 0 in PC p'
                    runs.append(self.swp[q % len(self.swp)](a, b, offa, offb, cw // WORD_SWAP)); q += 1
            for r in runs:
                r.wait()
            return
        # host-mediated exchange
        for p, pp in pairs:
            a, b = self.order[p], self.order[pp]
            self.st[a].sync(FROM, (1 << L) * AMP, 0); self.st[b].sync(FROM, (1 << L) * AMP, 0)
            va = self._view(a).reshape(nchunk, 2, 1 << v); vb = self._view(b).reshape(nchunk, 2, 1 << v)
            tmp = va[:, 1, :].copy(); va[:, 1, :] = vb[:, 0, :]; vb[:, 0, :] = tmp
            self.st[a].sync(TO, (1 << L) * AMP, 0); self.st[b].sync(TO, (1 << L) * AMP, 0)

    def _swap_local_local(self, a, b):
        L = self.L
        for i in range(self.P):
            self.st[i].sync(FROM, (1 << L) * AMP, 0)
            v = self._view(i)
            perm = list(range(L)); perm[a], perm[b] = b, a
            v[:] = host_remap(v.copy(), L, perm)
            self.st[i].sync(TO, (1 << L) * AMP, 0)

    def _remap(self, perm):
        L = self.L
        for a, b in transpositions(perm):
            if b < L:
                self._swap_local_local(a, b)
            elif a < L:
                self._swap_bank_local(b - L, a)
            else:   # bank <-> bank: relabel PCs, no data movement
                ja, jb = a - L, b - L
                new = list(self.order)
                for p in range(self.P):
                    bitsw = ((p >> ja) & 1) != ((p >> jb) & 1)
                    q = p ^ ((1 << ja) | (1 << jb)) if bitsw else p
                    new[p] = self.order[q]
                self.order = new

    # ------------------------------------------------------------------ public
    def run(self, qc, shots=1024, k=KMAX, p_max=None, fuse=True, tiling=True, initial_map=None,
            return_state=True, seed=None):
        from qiskit import QuantumCircuit
        n = qc.num_qubits
        P, n_run, L = self.plan(n, p_max)
        k = min(k, L)
        t0 = time.perf_counter()
        if n_run != n:
            q2 = QuantumCircuit(n_run); q2.compose(qc.remove_final_measurements(inplace=False), range(n), inplace=True); qrun = q2
        else:
            qrun = qc.remove_final_measurements(inplace=False)
        prog, fmap, stats = compile_circuit(qrun, L, k, fuse_1q=fuse, tiling=tiling, initial_map=initial_map, max_high=self.max_high)
        t1 = time.perf_counter()
        self._alloc(P, L)
        self.order = list(range(P))
        t1b = time.perf_counter()
        self._init_zero_state()
        t2 = time.perf_counter()
        T = dict(T_compile=t1 - t0, T_h2d=t2 - t1b, T_alloc=t1b - t1, T_kernel=0.0, T_exchange=0.0)
        started = False
        for kind, payload in prog:
            if kind == "remap" and not started:
                continue      # the state is still |0...0>, which every qubit permutation leaves unchanged
            if kind == "segment":
                started = True
                a = time.perf_counter(); self._upload_ins(payload); b = time.perf_counter()
                self._run_segment(); c = time.perf_counter()
                T["T_h2d"] += b - a; T["T_kernel"] += c - b
            else:
                a = time.perf_counter(); self._remap(payload); T["T_exchange"] += time.perf_counter() - a
        a = time.perf_counter()
        for i in range(P):
            self.st[i].sync(FROM, (1 << L) * AMP, 0)
        b = time.perf_counter()
        T["T_d2h"] = b - a
        counts = sample_slices([self._view(self.order[p]) for p in range(P)], shots, fmap, n_run, n, seed)
        c = time.perf_counter()
        T["T_post"] = c - b
        T["T_e2e"] = T["T_compile"] + T["T_h2d"] + T["T_kernel"] + T["T_exchange"] + T["T_d2h"] + T["T_post"]
        state = None
        if return_state:   # full logical-order state (fidelity checks); reported separately, not in T_e2e
            full = np.concatenate([self._view(self.order[p]) for p in range(P)])
            state = host_remap(full, n_run, [fmap[l] for l in range(n_run)])[: 1 << n]
            T["T_state_reorder"] = time.perf_counter() - c
        out = dict(timing=T, P=P, L=L, k=k, n_run=n_run, stats=stats.__dict__, counts=counts)
        if return_state:
            out["state"] = state
        return out
