"""Instruction set shared by the host compiler, the numpy golden model and the HLS kernel.

All records are int32 words; matrices are float32 bit patterns.

Stream layout (one kernel launch executes one segment = list of passes between host remaps):
  word 0          : MAGIC
  word 1          : L      (local qubits per PC: slice length 2^L amplitudes)
  word 2          : n_passes
  word 3          : k      (tile = 2^k amplitudes)
  words 4..7      : reserved
  then per pass   : PASS_WORDS header + n_ops * OP_WORDS op records

Pass header: [m, h0, h1, n_ops, h2, h3, 0, 0]
  m in {0..MAX_HIGH}: number of high local qubits gathered into the tile. Tile qubits 0..k-m-1 are physical
  qubits 0..k-m-1; tile qubit k-m+i is h_i. Kernels built with MAXH=2 ignore words 4, 5.

Op record (OP_WORDS = 24): [opcode=OP_MUX2, t, cmask, 0, 0, 0, 0, 0, f0..f15]
  t     : tile qubit the 2x2 acts on
  cmask : bit mask over tile qubits; a pair (i0, i1) along t uses M1 if cmask != 0 and
          (i0 & cmask) == cmask, else M0
  f0..f7 = M0 (m00 m01 m10 m11 as re, im), f8..f15 = M1
Gate mapping (host): U2 -> MUX2(t, 0, U, U); controlled-U -> MUX2(t, ctrls, I, U) (any number of
controls inside the tile window); 2q diagonal d -> MUX2(b, 1<<a, diag(d0, d2), diag(d1, d3));
any other 2q unitary -> 3 MUX2 via the cosine-sine decomposition.
"""
MAGIC = 0x51535632  # "QSV2"
HDR_WORDS = 8
PASS_WORDS = 8
OP_WORDS = 24
OP_MUX2 = 1
MAX_OPS_PER_PASS = 64
MAX_HIGH = 4      # default; must not exceed the kernel build's MAXH
