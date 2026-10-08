// HBM ceiling microbenchmark, one AXI port per compute unit (instantiated N times; same structure as sv_engine).
// A single 32-port kernel was not routable on the U55C (builds/hbm_stream*: "global congestion level 7").
//
// Runtime arguments:
//   mode 0 read (XOR-reduce, result to word 2n), 1 write, 3 copy (reader -> FIFO -> writer dataflow, dst at word n)
//   n    port words per repetition, reps repetitions
// Port width: HBM1_W (256 default, or 512).
#include <ap_int.h>
#include <hls_stream.h>
#ifndef HBM1_W
#define HBM1_W 256
#endif
typedef ap_uint<HBM1_W> pw;

static void rd(const pw *p, hls::stream<pw> &s, long n, int reps) {
    for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4194304
            s.write(p[i]);
        }
}
static void wr(pw *p, hls::stream<pw> &s, long n, int reps) {
    for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4194304
            p[n + i] = s.read();
        }
}
static void copy_df(pw *p, long n, int reps) {
#pragma HLS DATAFLOW
    hls::stream<pw, 64> s;
    rd(p, s, n, reps);
    wr(p, s, n, reps);
}

extern "C" void hbm1(pw *p, int mode, long n, int reps) {
#pragma HLS INTERFACE m_axi port = p bundle = gp offset = slave max_read_burst_length = 64 \
    max_write_burst_length = 64 num_read_outstanding = 16 num_write_outstanding = 16 latency = 64 depth = 1048576
#pragma HLS INTERFACE s_axilite port = return
    if (mode == 3) { copy_df(p, n, reps); return; }
    if (mode == 0) {
        pw acc = 0;
    RD: for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4194304
            acc ^= p[i];
        }
        p[2 * n] = acc;
    } else {
    WR: for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4194304
            p[i] = (pw)(i + r);
        }
    }
}
