"""Multi-card SV runtime: the statevector is split over C cards x P engine PCs = C*P slices.

Physical qubit layout: local qubits 0..L-1 inside a slice, then log2(P) PC bits, then log2(C) card bits (top).
The compiler is unchanged (it sees C*P "PCs"). Execution:
  segment : the same instruction stream is started on every engine CU of every card, then all are awaited
            (cards run concurrently; no host threads needed)
  remap   : bank<->local transpositions. Partner slices on the same card use that card's device swap kernel;
            partners on different cards are exchanged through host memory (D2H of the two chunk sets, swap,
            H2D), i.e. PCIe-bound. bank<->bank: relabel slices. local<->local: per-slice host permutation.
Timing fields as in runtime.py; T_exchange is split into exchange_device_s and exchange_host_s.
"""
import uuid  # noqa: F401
import os, sys, time
import numpy as np
import pyxrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import SVRuntime, transpositions, sample_slices, TO, FROM, AMP, WORD_SWAP, KMAX  # noqa: E402
LMIN = 7   # multi-card runs target large n; small slices are only used by the orchestration tests
from compiler import compile_circuit  # noqa: E402
from model import remap as host_remap  # noqa: E402


class MultiSVRuntime:
    def __init__(self, bdfs, xclbin, n_engines, n_swap, max_high=None):
        self.cards = [SVRuntime(b, xclbin, n_engines, n_swap, max_high=max_high) for b in bdfs]
        self.max_high = self.cards[0].max_high
        self.C = len(bdfs)
        assert self.C & (self.C - 1) == 0
        self.T_setup = sum(c.T_setup for c in self.cards)

    def plan(self, n, p_max=None):
        """Slices per card: one per engine when a slice fits one PC (preferred slice >= 2^LPREF amplitudes),
        virtual slices (more than one per engine) only when needed for capacity."""
        from runtime import LPREF
        card = self.cards[0]
        cap = p_max or card.n_slots
        lc = self.C.bit_length() - 1
        P = 1
        while P * 2 <= min(cap, len(card.eng)) and n - lc - (P * 2).bit_length() + 1 >= max(LPREF, LMIN):
            P *= 2
        fits = lambda P: (1 << (n - lc - P.bit_length() + 1)) * AMP + (4 << 20) <= (512 << 20)
        while not fits(P) and P * 2 <= cap:
            P *= 2
        L = n - lc - (P.bit_length() - 1)
        return P, L

    # slice s = card * P + pc ; order[s] = (card, bo index) holding logical slice s
    def _bo(self, s):
        c, i = self.order[s]
        return self.cards[c], i

    def _swap_bank_local(self, j, v):
        L = self.L
        cw = (1 << v) * AMP
        nchunk = 1 << (L - v - 1)
        S = self.C * self.P
        runs = []
        host_pairs = []
        for s in range(S):
            if (s >> j) & 1:
                continue
            s2 = s | (1 << j)
            (ca, ia), (cb, ib) = self.order[s], self.order[s2]
            if ca == cb and self.cards[ca].use_device_swap and self.cards[ca].swap_ok and cw % WORD_SWAP == 0:
                card = self.cards[ca]
                q = 0
                for c in range(nchunk):
                    runs.append(card.swp[q % len(card.swp)](card.st[ia], card.st[ib], ((2 * c + 1) * cw) // WORD_SWAP,
                                                            (2 * c * cw) // WORD_SWAP, cw // WORD_SWAP)); q += 1
            else:
                host_pairs.append(((ca, ia), (cb, ib)))
        t0 = time.perf_counter()
        for r in runs:
            r.wait()
        t1 = time.perf_counter()
        full = (1 << L) * AMP
        for (ca, ia), (cb, ib) in host_pairs:
            A, B = self.cards[ca], self.cards[cb]
            A.st[ia].sync(FROM, full, 0); B.st[ib].sync(FROM, full, 0)
            va = np.frombuffer(A.st[ia].map(), dtype=np.complex64, count=1 << L).reshape(nchunk, 2, 1 << v)
            vb = np.frombuffer(B.st[ib].map(), dtype=np.complex64, count=1 << L).reshape(nchunk, 2, 1 << v)
            tmp = va[:, 1, :].copy(); va[:, 1, :] = vb[:, 0, :]; vb[:, 0, :] = tmp
            A.st[ia].sync(TO, full, 0); B.st[ib].sync(TO, full, 0)
        t2 = time.perf_counter()
        self.T["exchange_device_s"] += t1 - t0
        self.T["exchange_host_s"] += t2 - t1

    def _remap(self, perm):
        L = self.L
        for a, b in transpositions(perm):
            if b < L:
                t0 = time.perf_counter()
                for card in self.cards:
                    card._swap_local_local(a, b)
                self.T["exchange_host_s"] += time.perf_counter() - t0
            elif a < L:
                self._swap_bank_local(b - L, a)
            else:
                ja, jb = a - L, b - L
                new = list(self.order)
                for s in range(len(self.order)):
                    if ((s >> ja) & 1) != ((s >> jb) & 1):
                        new[s] = self.order[s ^ ((1 << ja) | (1 << jb))]
                self.order = new

    def run(self, qc, shots=1024, k=KMAX, p_max=None, fuse=True, tiling=True, return_state=True, seed=None):
        n = qc.num_qubits
        P, L = self.plan(n, p_max)
        assert L >= LMIN, "circuit too small for this many cards"
        self.P, self.L = P, L
        k = min(k, L)
        t0 = time.perf_counter()
        prog, fmap, stats = compile_circuit(qc.remove_final_measurements(inplace=False), L, k, fuse_1q=fuse, tiling=tiling, max_high=self.max_high)
        t1 = time.perf_counter()
        for card in self.cards:
            card._alloc(P, L)
            card.order = list(range(P))
        for card in self.cards:          # device zeroing on every card (each card's CUs run concurrently)
            card._upload_ins(card._zero_segment())
        runs = [card.eng[card.slot[i][0]](card.st[i], (1 << L) * AMP // 32) for card in self.cards for i in range(P)]
        for r in runs:
            r.wait()
        v = self.cards[0]._view(0); v[:4] = 0; v[0] = 1
        self.cards[0].st[0].sync(TO, 32, 0)
        self.order = [(c, i) for c in range(self.C) for i in range(P)]
        t2 = time.perf_counter()
        self.T = dict(T_compile=t1 - t0, T_h2d=t2 - t1, T_kernel=0.0, exchange_device_s=0.0, exchange_host_s=0.0)
        started = False
        for kind, payload in prog:
            if kind == "remap" and not started:
                continue
            if kind == "segment":
                started = True
                a = time.perf_counter()
                for card in self.cards:
                    card._upload_ins(payload)
                b = time.perf_counter()
                ins_off = (1 << L) * AMP // 32
                runs = [card.eng[card.slot[i][0]](card.st[i], ins_off) for card in self.cards for i in range(P)]
                for r in runs:
                    r.wait()
                c = time.perf_counter()
                self.T["T_h2d"] += b - a; self.T["T_kernel"] += c - b
            else:
                self._remap(payload)
        self.T["T_exchange"] = self.T["exchange_device_s"] + self.T["exchange_host_s"]
        a = time.perf_counter()
        for card in self.cards:
            for i in range(P):
                card.st[i].sync(FROM, (1 << L) * AMP, 0)
        b = time.perf_counter()
        self.T["T_d2h"] = b - a
        slices = [self._bo(s)[0]._view(self._bo(s)[1]) for s in range(self.C * P)]
        counts = sample_slices(slices, shots, fmap, n, n, seed)
        c = time.perf_counter()
        self.T["T_post"] = c - b
        self.T["T_e2e"] = sum(self.T[x] for x in ("T_compile", "T_h2d", "T_kernel", "T_exchange", "T_d2h", "T_post"))
        state = None
        if return_state:
            state = host_remap(np.concatenate(slices), n, [fmap[l] for l in range(n)])
            self.T["T_state_reorder"] = time.perf_counter() - c
        out = dict(timing=dict(self.T), C=self.C, P=P, L=L, k=k, stats=stats.__dict__, counts=counts)
        if return_state:
            out["state"] = state
        return out
