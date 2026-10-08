// Device-side block swap between two buffers that may live in different HBM pseudo-channels
// (both ports are connected to HBM[0:31] through the HBM switch). Used for bank-qubit remaps:
// swapping bank qubit j with local qubit v exchanges, for each PC pair, the contiguous
// 2^v-amplitude chunks; with v = L-1 each PC exchanges one half-slice.
// Sequential burst phases per block (read A, read B, write A, write B), so no RAW hazard is possible.
#include <ap_int.h>
typedef ap_uint<512> w512;
#define BLK 1024   // words per block (64 KB)

extern "C" void sv_swap(w512 *a, w512 *b, long offa, long offb, long nwords) {
#pragma HLS INTERFACE m_axi port = a bundle = ga offset = slave max_read_burst_length = 64 \
    max_write_burst_length = 64 num_read_outstanding = 16 num_write_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE m_axi port = b bundle = gb offset = slave max_read_burst_length = 64 \
    max_write_burst_length = 64 num_read_outstanding = 16 num_write_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE s_axilite port = return
    w512 ba[BLK], bb[BLK];
#pragma HLS BIND_STORAGE variable = ba type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = bb type = ram_2p impl = uram
BLOCKS:
    for (long s = 0; s < nwords; s += BLK) {
        const int n = (nwords - s) < BLK ? (int)(nwords - s) : BLK;
    RA: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
            ba[i] = a[offa + s + i];
        }
    RB: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
            bb[i] = b[offb + s + i];
        }
    WA: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
            a[offa + s + i] = bb[i];
        }
    WB: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
            b[offb + s + i] = ba[i];
        }
    }
}
