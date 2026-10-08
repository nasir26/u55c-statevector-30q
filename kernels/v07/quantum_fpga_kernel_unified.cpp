#include <ap_int.h>
#include <hls_math.h>

/*
 * ===================================================================================
 * UNIFIED HBM-PARALLEL / II=1 QUANTUM STATEVECTOR KERNEL (Vitis HLS)
 * Target: AMD Alveo U55C (xilinx_u55c_gen3x4_xdma_base_3_202210_1)
 * ===================================================================================
 *
 * This file merges two prior designs:
 *
 *   - quantum_fpga_kernel_complex128_v06.cpp: 8-way HBM bank interleaving,
 *     giving genuine parallel-channel bandwidth and letting each bank hold
 *     only N/8 of the statevector (needed to fit 26-27+ qubits in HBM2).
 *
 *   - opt_kernel.cpp: `#pragma HLS DEPENDENCE ... type=inter false` to reach
 *     II=1, and branch-free index reconstruction for CX/CZ so the loop trip
 *     count drops to exactly the number of amplitudes that actually change
 *     (N/2 for CX, N/4 for CZ) instead of looping over all N with an `if`.
 *
 * The bank-select switch in read_amp/write_amp is retained deliberately:
 * which of the 8 banks a given global index lives in is a runtime address
 * bit (idx & 0x7), not something resolvable at HLS compile time, so a
 * multiplexer of some form is unavoidable if you want 8-way HBM capacity
 * scaling. What was missing in v06 was the DEPENDENCE override, which is
 * the actual reason it fell back to II=16/II=8 rather than the switch
 * itself. Adding that override, on all 8 bank pointers, combined with
 * opt_kernel's branch-free loops, is what lets this version pipeline at
 * II=1 while keeping 8-way parallel HBM access.
 * ===================================================================================
 */

// ── Opcode constants ─────────────────────────────────────────
#define GATE_ID    0
#define GATE_H     1
#define GATE_X     2
#define GATE_Y     3
#define GATE_Z     4
#define GATE_S     5
#define GATE_T     6
#define GATE_SDG   7
#define GATE_TDG   8
#define GATE_SX    9
#define GATE_SXDG  10
#define GATE_RX    11
#define GATE_RY    12
#define GATE_RZ    13
#define GATE_P     14
#define GATE_U2    15
#define GATE_U3    16
#define GATE_CX    17
#define GATE_CZ    18

// ── Bank layout ───────────────────────────────────────────────
// Global amplitude index i -> bank = i & 0x7, slot = i >> 3.
// Each slot stores [re, im] as two consecutive doubles, so each
// bank needs (2^num_qubits / 8) * 16 bytes of HBM capacity.
#define NUM_BANKS  8
#define BANK_MASK  0x7
#define BANK_SHIFT 3

#define BANK_PARAMS \
    double* b0, double* b1, double* b2, double* b3, \
    double* b4, double* b5, double* b6, double* b7

#define BANK_ARGS \
    b0, b1, b2, b3, b4, b5, b6, b7

#define BANK_ARGS_K \
    bank0, bank1, bank2, bank3, bank4, bank5, bank6, bank7

// Tell the scheduler every bank pointer is free of cross-iteration
// hazards within the calling loop. Must be issued once per bank per
// loop that touches it -- this is the piece v06 was missing.
#define DECLARE_BANK_INDEPENDENCE \
    _Pragma("HLS DEPENDENCE variable=b0 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b1 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b2 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b3 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b4 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b5 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b6 type=inter false") \
    _Pragma("HLS DEPENDENCE variable=b7 type=inter false")

// ── Amplitude I/O ─────────────────────────────────────────────
inline void read_amp(int idx, double &re, double &im, BANK_PARAMS) {
    #pragma HLS INLINE
    int bank = idx & BANK_MASK;
    int off  = (idx >> BANK_SHIFT) << 1;
    double *p;
    switch (bank) {
        case 0: p = b0; break; case 1: p = b1; break;
        case 2: p = b2; break; case 3: p = b3; break;
        case 4: p = b4; break; case 5: p = b5; break;
        case 6: p = b6; break; default: p = b7; break;
    }
    re = p[off]; im = p[off + 1];
}

inline void write_amp(int idx, double re, double im, BANK_PARAMS) {
    #pragma HLS INLINE
    int bank = idx & BANK_MASK;
    int off  = (idx >> BANK_SHIFT) << 1;
    double *p;
    switch (bank) {
        case 0: p = b0; break; case 1: p = b1; break;
        case 2: p = b2; break; case 3: p = b3; break;
        case 4: p = b4; break; case 5: p = b5; break;
        case 6: p = b6; break; default: p = b7; break;
    }
    p[off] = re; p[off + 1] = im;
}

inline void cmul(double ar, double ai, double br, double bi,
                 double &rr, double &ri) {
    #pragma HLS INLINE
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}

// ============================================================
//  FUNCTION 1 — apply_single_gate  (II=1, 8-bank parallel)
// ============================================================
static void apply_single_gate(
    BANK_PARAMS,
    int tgt, int nq,
    double g00r, double g00i,
    double g01r, double g01i,
    double g10r, double g10i,
    double g11r, double g11i
) {
    #pragma HLS INLINE off
    const int pairs  = 1 << (nq - 1);
    const int stride = 1 << tgt;

    SQ_LOOP: for (int p = 0; p < pairs; p++) {
        #pragma HLS PIPELINE II=1
        DECLARE_BANK_INDEPENDENCE
        #pragma HLS LOOP_TRIPCOUNT min=1 max=268435456

        int i0 = ((p >> tgt) << (tgt + 1)) | (p & (stride - 1));
        int i1 = i0 + stride;

        double a0r, a0i, a1r, a1i;
        read_amp(i0, a0r, a0i, BANK_ARGS);
        read_amp(i1, a1r, a1i, BANK_ARGS);

        double t0r, t0i, t1r, t1i;
        cmul(g00r, g00i, a0r, a0i, t0r, t0i);
        cmul(g01r, g01i, a1r, a1i, t1r, t1i);
        double n0r = t0r + t1r, n0i = t0i + t1i;

        cmul(g10r, g10i, a0r, a0i, t0r, t0i);
        cmul(g11r, g11i, a1r, a1i, t1r, t1i);
        double n1r = t0r + t1r, n1i = t0i + t1i;

        write_amp(i0, n0r, n0i, BANK_ARGS);
        write_amp(i1, n1r, n1i, BANK_ARGS);
    }
}

// ============================================================
//  FUNCTION 2 — apply_cnot  (branch-free, N/2 iterations, II=1)
// ============================================================
static void apply_cnot(BANK_PARAMS, int ctrl, int tgt, int nq) {
    #pragma HLS INLINE off
    const int N_half = 1 << (nq - 1);
    int min_bit = (ctrl < tgt) ? ctrl : tgt;
    int max_bit = (ctrl > tgt) ? ctrl : tgt;

    CX_LOOP: for (int p = 0; p < N_half; p++) {
        #pragma HLS PIPELINE II=1
        DECLARE_BANK_INDEPENDENCE
        #pragma HLS LOOP_TRIPCOUNT min=1 max=268435456

        int p_low  = p & ((1 << min_bit) - 1);
        int p_mid  = (p >> min_bit) & ((1 << (max_bit - min_bit - 1)) - 1);
        int p_high = (p >> (max_bit - 1));

        int i = p_low | (p_mid << (min_bit + 1)) | (p_high << (max_bit + 1)) | (1 << ctrl);
        int j = i ^ (1 << tgt);

        double ar, ai, br, bi;
        read_amp(i, ar, ai, BANK_ARGS);
        read_amp(j, br, bi, BANK_ARGS);
        write_amp(i, br, bi, BANK_ARGS);
        write_amp(j, ar, ai, BANK_ARGS);
    }
}

// ============================================================
//  FUNCTION 3 — apply_cz  (branch-free, N/4 iterations, II=1)
// ============================================================
static void apply_cz(BANK_PARAMS, int ctrl, int tgt, int nq) {
    #pragma HLS INLINE off
    const int N_quarter = 1 << (nq - 2);
    int min_bit = (ctrl < tgt) ? ctrl : tgt;
    int max_bit = (ctrl > tgt) ? ctrl : tgt;

    CZ_LOOP: for (int p = 0; p < N_quarter; p++) {
        #pragma HLS PIPELINE II=1
        DECLARE_BANK_INDEPENDENCE
        #pragma HLS LOOP_TRIPCOUNT min=1 max=134217728

        int p_low  = p & ((1 << min_bit) - 1);
        int p_mid  = (p >> min_bit) & ((1 << (max_bit - min_bit - 1)) - 1);
        int p_high = (p >> (max_bit - 1));

        int i = p_low | (p_mid << (min_bit + 1)) | (p_high << (max_bit + 1))
                | (1 << ctrl) | (1 << tgt);

        double re, im;
        read_amp(i, re, im, BANK_ARGS);
        write_amp(i, -re, -im, BANK_ARGS);
    }
}

// ============================================================
//  TOP-LEVEL KERNEL
// ============================================================
extern "C" void quantum_fpga_kernel_complex128(
    double* bank0, double* bank1, double* bank2, double* bank3,
    double* bank4, double* bank5, double* bank6, double* bank7,
    int*    gate_sequence,
    int     num_gates,
    int     num_qubits
) {
    #pragma HLS INTERFACE m_axi port=bank0 max_widen_bitwidth=512 offset=slave bundle=bank0
    #pragma HLS INTERFACE m_axi port=bank1 max_widen_bitwidth=512 offset=slave bundle=bank1
    #pragma HLS INTERFACE m_axi port=bank2 max_widen_bitwidth=512 offset=slave bundle=bank2
    #pragma HLS INTERFACE m_axi port=bank3 max_widen_bitwidth=512 offset=slave bundle=bank3
    #pragma HLS INTERFACE m_axi port=bank4 max_widen_bitwidth=512 offset=slave bundle=bank4
    #pragma HLS INTERFACE m_axi port=bank5 max_widen_bitwidth=512 offset=slave bundle=bank5
    #pragma HLS INTERFACE m_axi port=bank6 max_widen_bitwidth=512 offset=slave bundle=bank6
    #pragma HLS INTERFACE m_axi port=bank7 max_widen_bitwidth=512 offset=slave bundle=bank7
    #pragma HLS INTERFACE m_axi port=gate_sequence offset=slave bundle=gmem_instr

    #pragma HLS INTERFACE s_axilite port=bank0         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank1         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank2         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank3         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank4         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank5         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank6         bundle=control
    #pragma HLS INTERFACE s_axilite port=bank7         bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_sequence bundle=control
    #pragma HLS INTERFACE s_axilite port=num_gates     bundle=control
    #pragma HLS INTERFACE s_axilite port=num_qubits    bundle=control
    #pragma HLS INTERFACE s_axilite port=return        bundle=control

    // ── Load gate sequence into BRAM ──────────────────────────
    int gbuf[512 * 8];
    #pragma HLS BIND_STORAGE variable=gbuf type=ram_2p impl=bram

    int total = num_gates * 8;
    LOAD: for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8 max=4096
        gbuf[i] = gate_sequence[i];
    }

    // ── Execute gates ─────────────────────────────────────────
    EXEC: for (int g = 0; g < num_gates; g++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512

        int base = g * 8;
        int op   = gbuf[base];
        int q0   = gbuf[base + 1];
        int q1   = gbuf[base + 2];

        union { int i; float f; } p0, p1, p2;
        p0.i = gbuf[base + 4];
        p1.i = gbuf[base + 5];
        p2.i = gbuf[base + 6];
        double dp0 = (double)p0.f;
        double dp1 = (double)p1.f;
        double dp2 = (double)p2.f;

        if (op == GATE_ID) {
            /* no-op */

        } else if (op == GATE_H) {
            const double h = 0.7071067811865476;
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                 h, 0.0,  h, 0.0,
                 h, 0.0, -h, 0.0);

        } else if (op == GATE_X) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0);

        } else if (op == GATE_Y) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                0.0,  0.0, 0.0, -1.0,
                0.0,  1.0, 0.0,  0.0);

        } else if (op == GATE_Z) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0,  0.0, 0.0,
                0.0, 0.0, -1.0, 0.0);

        } else if (op == GATE_S) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0);

        } else if (op == GATE_T) {
            const double t = 0.7071067811865476;
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   t,   t);

        } else if (op == GATE_SDG) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0, 0.0,  0.0,
                0.0, 0.0, 0.0, -1.0);

        } else if (op == GATE_TDG) {
            const double t = 0.7071067811865476;
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   t,  -t);

        } else if (op == GATE_SX) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                0.5,  0.5,  0.5, -0.5,
                0.5, -0.5,  0.5,  0.5);

        } else if (op == GATE_SXDG) {
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                0.5, -0.5,  0.5,  0.5,
                0.5,  0.5,  0.5, -0.5);

        } else if (op == GATE_RX) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                  c, 0.0, 0.0,  -s,
                0.0,  -s,   c, 0.0);

        } else if (op == GATE_RY) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                c, 0.0, -s, 0.0,
                s, 0.0,  c, 0.0);

        } else if (op == GATE_RZ) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                c, -s, 0.0, 0.0,
                0.0, 0.0, c,  s);

        } else if (op == GATE_P) {
            double c = hls::cos(dp0);
            double s = hls::sin(dp0);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   c,   s);

        } else if (op == GATE_U2) {
            const double inv_sqrt2 = 0.7071067811865476;
            double cphi = hls::cos(dp0), sphi = hls::sin(dp0);
            double clam = hls::cos(dp1), slam = hls::sin(dp1);
            double cpl  = hls::cos(dp0 + dp1);
            double spl  = hls::sin(dp0 + dp1);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                 inv_sqrt2,              0.0,
                -inv_sqrt2 * clam, -inv_sqrt2 * slam,
                 inv_sqrt2 * cphi,  inv_sqrt2 * sphi,
                 inv_sqrt2 * cpl,   inv_sqrt2 * spl);

        } else if (op == GATE_U3) {
            double ct   = hls::cos(dp0 * 0.5), st = hls::sin(dp0 * 0.5);
            double cphi = hls::cos(dp1),       sphi = hls::sin(dp1);
            double clam = hls::cos(dp2),       slam = hls::sin(dp2);
            double cpl  = hls::cos(dp1 + dp2);
            double spl  = hls::sin(dp1 + dp2);
            apply_single_gate(BANK_ARGS_K, q0, num_qubits,
                  ct,          0.0,
                -st * clam,  -st * slam,
                 st * cphi,   st * sphi,
                 ct * cpl,    ct * spl);

        } else if (op == GATE_CX) {
            apply_cnot(BANK_ARGS_K, q0, q1, num_qubits);

        } else if (op == GATE_CZ) {
            apply_cz(BANK_ARGS_K, q0, q1, num_qubits);
        }
    }
}