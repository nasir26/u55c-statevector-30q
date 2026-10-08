// SV engine: one compute unit per HBM pseudo-channel (instantiated N times with nk=sv_engine:N).
// Executes one segment of the QSV2 instruction stream (host/sv/isa.py) on the PC's statevector slice.
//
// Data: complex64, 4 amplitudes per 256-bit word (lane l: re = bits[64l, 64l+32), im = bits[64l+32, 64l+64)),
// i.e. the numpy complex64 memory layout.
//
// Per pass: DATAFLOW  reader --(tile blocks, 2 banks)--> compute --(tile blocks)--> writer
//   - tile = 2^k amplitudes = 2^(k-2) words; a pass may gather m <= 2 high local qubits h0, h1, so the
//     tile is 2^m sub-blocks of 2^(k-m-2) contiguous words (each sub-block is one long burst).
//   - tile word w is stored in bank popcount(w) & 1 at address w >> 1. Words whose indices differ in
//     exactly one bit are always in different banks, so every pair update reads one word from each bank
//     and writes one word to each bank per cycle (II = 1, 4 amplitude pairs per cycle).
//   - op MUX2(t, cmask, M0, M1): a pair (i0, i1 = i0 | 1<<t) uses M1 if cmask != 0 and all cmask bits
//     of i0 are set, else M0. The first op reads the input block, the last op writes the output block,
//     intermediate results live in a private two-bank working buffer (output blocks are write-only).
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>

#define KMAX 12
#define NW (1 << (KMAX - 2))   // words per tile (max)
#define HALF (NW / 2)
#define MAXOPS 64
#define OPW 24
#define PASSW 8
#define HDRW 8

typedef ap_uint<256> w256;
typedef w256 bank_t[HALF];
// SV_WIDE: 512-bit AXI port; reader/writer move two internal 256-bit words per cycle (words 2j and 2j+1 have
// opposite parity, so each beat puts exactly one word in each bank). Default: 256-bit port, one word per cycle.
#ifdef SV_WIDE
typedef ap_uint<512> pw_t;
#define PW 2
#else
typedef w256 pw_t;
#define PW 1
#endif

static inline float f_of(ap_uint<32> u) {
    union { unsigned int i; float f; } c; c.i = (unsigned int)u; return c.f;
}
static inline ap_uint<32> u_of(float f) {
    union { unsigned int i; float f; } c; c.f = f; return (ap_uint<32>)c.i;
}
static inline int parity12(int w) {
#pragma HLS INLINE
    ap_uint<KMAX> x = w;
    return x.xor_reduce();
}

#ifndef MAXH
#define MAXH 4          // max high local qubits gathered per pass (2^MAXH sub-blocks per tile)
#endif
struct PassCfg {
    int L, k, m, h[4];
};

// bit mask (over local amplitude-index bits) of the positions enumerated by the tile counter.
// Tile t's base index is pdep(t, freemask); successive bases follow next = ((base | ~fm) + 1) & fm.
static inline ap_uint<32> free_mask(const PassCfg &c) {
#pragma HLS INLINE
    ap_uint<32> fm = 0;
    for (int b = 0; b < 32; b++) {
#pragma HLS UNROLL
        bool used = (b < c.k - c.m) || (b >= c.L);
        for (int i = 0; i < MAXH; i++) {
#pragma HLS UNROLL
            used = used || (i < c.m && b == c.h[i]);
        }
        fm[b] = used ? 0 : 1;
    }
    return fm;
}
// amplitude offset of sub-block s: bit i of s selects high qubit h[i]
static inline long sub_off(const PassCfg &c, int s) {
#pragma HLS INLINE
    long o = 0;
    for (int i = 0; i < MAXH; i++) {
#pragma HLS UNROLL
        if (i < c.m && ((s >> i) & 1)) o |= (1L << c.h[i]);
    }
    return o;
}

static void reader(const pw_t *st, PassCfg c, hls::stream_of_blocks<bank_t> &q0,
                   hls::stream_of_blocks<bank_t> &q1) {
    const int ntiles = 1 << (c.L - c.k);
    const int lsb = c.k - c.m - 2;            // log2(256-bit words per sub-block)
    const int sbw = (1 << lsb) / PW;          // port words per sub-block
    const int nsub = 1 << c.m;
    const ap_uint<32> fm = free_mask(c);
    ap_uint<32> base = 0;
TILES:
    for (int t = 0; t < ntiles; t++) {
        hls::write_lock<bank_t> b0(q0), b1(q1);
    SUBS:
        for (int s = 0; s < nsub; s++) {
            const long wa = (((long)base | sub_off(c, s)) >> 2) / PW;
        WORDS:
            for (int j = 0; j < sbw; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 128 max = 1024
                const pw_t v = st[wa + j];
#ifdef SV_WIDE
                const int w = (s << lsb) | (2 * j);       // even word; w + 1 is in the other bank, same address
                const w256 lo = v.range(255, 0), hi = v.range(511, 256);
                if (parity12(w)) { b1[w >> 1] = lo; b0[w >> 1] = hi; }
                else             { b0[w >> 1] = lo; b1[w >> 1] = hi; }
#else
                const int w = (s << lsb) | j;
                if (parity12(w)) b1[w >> 1] = v; else b0[w >> 1] = v;
#endif
            }
        }
        base = ((base | ~fm) + 1) & fm;
    }
}

static void writer(pw_t *st, PassCfg c, hls::stream_of_blocks<bank_t> &q0,
                   hls::stream_of_blocks<bank_t> &q1) {
    const int ntiles = 1 << (c.L - c.k);
    const int lsb = c.k - c.m - 2;
    const int sbw = (1 << lsb) / PW;
    const int nsub = 1 << c.m;
    const ap_uint<32> fm = free_mask(c);
    ap_uint<32> base = 0;
TILES:
    for (int t = 0; t < ntiles; t++) {
        hls::read_lock<bank_t> b0(q0), b1(q1);
    SUBS:
        for (int s = 0; s < nsub; s++) {
            const long wa = (((long)base | sub_off(c, s)) >> 2) / PW;
        WORDS:
            for (int j = 0; j < sbw; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 128 max = 1024
#ifdef SV_WIDE
                const int w = (s << lsb) | (2 * j);
                pw_t v;
                if (parity12(w)) { v.range(255, 0) = b1[w >> 1]; v.range(511, 256) = b0[w >> 1]; }
                else             { v.range(255, 0) = b0[w >> 1]; v.range(511, 256) = b1[w >> 1]; }
                st[wa + j] = v;
#else
                const int w = (s << lsb) | j;
                st[wa + j] = parity12(w) ? b1[w >> 1] : b0[w >> 1];
#endif
            }
        }
        base = ((base | ~fm) + 1) & fm;
    }
}

// one pair update with matrix M (8 floats: m00 m01 m10 m11 as re, im)
static inline void pair2(float x0r, float x0i, float x1r, float x1i, const float M[8],
                         float &y0r, float &y0i, float &y1r, float &y1i) {
#pragma HLS INLINE
    y0r = M[0] * x0r - M[1] * x0i + M[2] * x1r - M[3] * x1i;
    y0i = M[0] * x0i + M[1] * x0r + M[2] * x1i + M[3] * x1r;
    y1r = M[4] * x0r - M[5] * x0i + M[6] * x1r - M[7] * x1i;
    y1i = M[4] * x0i + M[5] * x0r + M[6] * x1i + M[7] * x1r;
}

static void compute(PassCfg c, int nops, const int opt[MAXOPS], const int opc[MAXOPS],
                    const float opm[MAXOPS][16], hls::stream_of_blocks<bank_t> &i0q,
                    hls::stream_of_blocks<bank_t> &i1q, hls::stream_of_blocks<bank_t> &o0q,
                    hls::stream_of_blocks<bank_t> &o1q) {
    const int ntiles = 1 << (c.L - c.k);
    const int nwh = 1 << (c.k - 3);           // iterations per op = words / 2
    w256 wk0[HALF], wk1[HALF];
#pragma HLS BIND_STORAGE variable = wk0 type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = wk1 type = ram_t2p impl = bram
TILES:
    for (int t = 0; t < ntiles; t++) {
        hls::read_lock<bank_t> in0(i0q), in1(i1q);
        hls::write_lock<bank_t> out0(o0q), out1(o1q);
    OPS:
        for (int o = 0; o < nops; o++) {
            // Op o+1 reads words written by op o: the PAIRS pipeline must drain between ops. Without this,
            // HLS flattens OPS x PAIRS into one pipeline and (with the DEPENDENCE pragmas below) op o+1 reads
            // stale words for small tiles (observed on hardware for k <= 8; csim is sequential and cannot show it).
#pragma HLS LOOP_FLATTEN off
            const int tq = opt[o], cm = opc[o];
            float M0[8], M1[8];
#pragma HLS ARRAY_PARTITION variable = M0 complete
#pragma HLS ARRAY_PARTITION variable = M1 complete
            for (int e = 0; e < 8; e++) {
#pragma HLS UNROLL
                M0[e] = opm[o][e]; M1[e] = opm[o][8 + e];
            }
            const bool first = (o == 0);
            const bool last = (o == nops - 1);
            const int sb = tq >= 2 ? tq - 2 : 0;
        PAIRS:
            for (int i = 0; i < nwh; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512
#pragma HLS DEPENDENCE variable = wk0 type = inter false
#pragma HLS DEPENDENCE variable = wk1 type = inter false
                // the two words touched this iteration
                int wa, wb;
                if (tq >= 2) {
                    wa = ((i >> sb) << (sb + 1)) | (i & ((1 << sb) - 1));
                    wb = wa | (1 << sb);
                } else {
                    wa = 2 * i; wb = 2 * i + 1;
                }
                const int pa = parity12(wa);   // bank of wa; wb is in the other bank
                const int a0 = pa ? (wb >> 1) : (wa >> 1);   // address read from bank 0
                const int a1 = pa ? (wa >> 1) : (wb >> 1);   // address read from bank 1
                const w256 r0 = first ? in0[a0] : wk0[a0];
                const w256 r1 = first ? in1[a1] : wk1[a1];
                const w256 A = pa ? r1 : r0;   // word wa
                const w256 B = pa ? r0 : r1;   // word wb
                // unpack both words with constant lane slices
                float aR[4], aI[4], bR[4], bI[4];
                for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
                    aR[l] = f_of(A.range(64 * l + 31, 64 * l)); aI[l] = f_of(A.range(64 * l + 63, 64 * l + 32));
                    bR[l] = f_of(B.range(64 * l + 31, 64 * l)); bI[l] = f_of(B.range(64 * l + 63, 64 * l + 32));
                }
                // pair p: mode 2 (t>=2): lane p of A with lane p of B
                //         t=0: pairs (A0,A1) (A2,A3) (B0,B1) (B2,B3);  t=1: (A0,A2) (A1,A3) (B0,B2) (B1,B3)
                const int mode = tq >= 2 ? 2 : tq;
                float x0r[4], x0i[4], x1r[4], x1i[4];
                int ix0[4];
                for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    const int q = p & 1;
                    const bool inB = p >= 2;
                    const int l0t0 = 2 * q, l1t0 = 2 * q + 1, l0t1 = q, l1t1 = q + 2;
                    const float *SR = inB ? bR : aR, *SI = inB ? bI : aI;
                    if (mode == 2) {
                        x0r[p] = aR[p]; x0i[p] = aI[p]; x1r[p] = bR[p]; x1i[p] = bI[p]; ix0[p] = wa * 4 + p;
                    } else if (mode == 0) {
                        x0r[p] = SR[l0t0]; x0i[p] = SI[l0t0]; x1r[p] = SR[l1t0]; x1i[p] = SI[l1t0];
                        ix0[p] = (inB ? wb : wa) * 4 + l0t0;
                    } else {
                        x0r[p] = SR[l0t1]; x0i[p] = SI[l0t1]; x1r[p] = SR[l1t1]; x1i[p] = SI[l1t1];
                        ix0[p] = (inB ? wb : wa) * 4 + l0t1;
                    }
                }
                float y0r[4], y0i[4], y1r[4], y1i[4];
                for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    const bool sel = cm != 0 && ((ix0[p] & cm) == cm);
                    float Ms[8];
                    for (int e = 0; e < 8; e++) {
#pragma HLS UNROLL
                        Ms[e] = sel ? M1[e] : M0[e];
                    }
                    pair2(x0r[p], x0i[p], x1r[p], x1i[p], Ms, y0r[p], y0i[p], y1r[p], y1i[p]);
                }
                // scatter with constant lane slices: value for lane l of the new A and B words
                w256 NA, NB;
                for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
                    float ar, ai, br, bi;
                    if (mode == 2) {
                        ar = y0r[l]; ai = y0i[l]; br = y1r[l]; bi = y1i[l];
                    } else if (mode == 0) {      // lane l belongs to pair l/2, even lane = x0
                        const int q = l >> 1; const bool odd = l & 1;
                        ar = odd ? y1r[q] : y0r[q];         ai = odd ? y1i[q] : y0i[q];
                        br = odd ? y1r[2 + q] : y0r[2 + q]; bi = odd ? y1i[2 + q] : y0i[2 + q];
                    } else {                     // lane l belongs to pair l&1, lanes 2,3 = x1
                        const int q = l & 1; const bool up = l >= 2;
                        ar = up ? y1r[q] : y0r[q];          ai = up ? y1i[q] : y0i[q];
                        br = up ? y1r[2 + q] : y0r[2 + q];  bi = up ? y1i[2 + q] : y0i[2 + q];
                    }
                    NA.range(64 * l + 31, 64 * l) = u_of(ar); NA.range(64 * l + 63, 64 * l + 32) = u_of(ai);
                    NB.range(64 * l + 31, 64 * l) = u_of(br); NB.range(64 * l + 63, 64 * l + 32) = u_of(bi);
                }
                const w256 n0 = pa ? NB : NA, n1 = pa ? NA : NB;
                if (last) { out0[a0] = n0; out1[a1] = n1; }
                else      { wk0[a0] = n0;  wk1[a1] = n1; }
            }
        }
    }
}

static void run_pass(pw_t *st, PassCfg c, int nops, const int opt[MAXOPS], const int opc[MAXOPS],
                     const float opm[MAXOPS][16]) {
#pragma HLS DATAFLOW
    hls::stream_of_blocks<bank_t, 2> qi0, qi1, qo0, qo1;
    reader(st, c, qi0, qi1);
    compute(c, nops, opt, opc, opm, qi0, qi1, qo0, qo1);
    writer(st, c, qo0, qo1);
}

// 32-bit instruction word pos of the segment stored from word offset ins_off of the state buffer
// ins_off is always given in 256-bit words (host side is the same for both port widths)
static inline int insw(const pw_t *st, long ins_off, int pos) {
#pragma HLS INLINE
    const int per = 8 * PW;                    // 32-bit words per port word
    const pw_t v = st[ins_off / PW + pos / per];
    const int l = pos % per;
    return (int)(unsigned int)v.range(32 * l + 31, 32 * l);
}

// One AXI port per CU (the HBM subsystem has 33 port connections): the instruction segment is stored
// in the same buffer as the state slice, starting at 256-bit word ins_off.
extern "C" void sv_engine(pw_t *st, long ins_off) {
#pragma HLS INTERFACE m_axi port = st bundle = gst offset = slave max_read_burst_length = 32 \
    max_write_burst_length = 32 num_read_outstanding = 8 num_write_outstanding = 8 latency = 64 depth = 1048576
#pragma HLS INTERFACE s_axilite port = return
    const int L = insw(st, ins_off, 1), npass = insw(st, ins_off, 2), k = insw(st, ins_off, 3);
    int pos = HDRW;
PASSES:
    for (int p = 0; p < npass; p++) {
        PassCfg c;
        c.L = L; c.k = k; c.m = insw(st, ins_off, pos);
        c.h[0] = insw(st, ins_off, pos + 1); c.h[1] = insw(st, ins_off, pos + 2);
        c.h[2] = insw(st, ins_off, pos + 4); c.h[3] = insw(st, ins_off, pos + 5);   // pass header words 4, 5
        const int nops = insw(st, ins_off, pos + 3);
        pos += PASSW;
        int opt[MAXOPS], opc[MAXOPS];
        float opm[MAXOPS][16];
#pragma HLS ARRAY_PARTITION variable = opm dim = 2 complete
    LOADOPS:
        for (int o = 0; o < nops; o++) {
            opt[o] = insw(st, ins_off, pos + 1);
            opc[o] = insw(st, ins_off, pos + 2);
            for (int e = 0; e < 16; e++) {
#pragma HLS PIPELINE II = 1
                opm[o][e] = f_of((ap_uint<32>)(unsigned int)insw(st, ins_off, pos + 8 + e));
            }
            pos += OPW;
        }
        run_pass(st, c, nops, opt, opc, opm);
    }
}
