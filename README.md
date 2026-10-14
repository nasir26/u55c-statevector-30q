# FPGA Statevector Simulator: 30 Qubits on one Alveo U55C

A statevector quantum-circuit simulator for the AMD/Xilinx Alveo U55C (HBM2) FPGA, up to 30 qubits on one card,
together with the CPU and GPU baselines, raw measurements and analysis scripts used to evaluate it.

The design compiles every gate to a single MUX2 primitive (one 2x2 complex matrix applied to amplitude pairs,
optionally controlled), schedules gates into passes that share one sweep over HBM, and runs one engine compute
unit per HBM pseudo-channel, with a device-side exchange kernel (`sv_swap`) for qubit remapping between channels.

## Layout

| Path | What |
|---|---|
| `kernels/sv/` | HLS sources of the engine (`sv_engine.cpp`) and exchange (`sv_swap.cpp`) kernels, C testbench |
| `kernels/hbm_stream/` | HBM streaming microbenchmark kernel |
| `kernels/v07/`, `host/v07/` | earlier (v07) kernel and host code, kept as the baseline; see `docs/PROVENANCE.md` |
| `host/sv/` | ISA, compiler (Qiskit circuit -> MUX2 records), golden model (NumPy), XRT runtime |
| `builds/<name>/` | link configuration, `sv.xclbin` hashes and post-route reports of each bitstream |
| `tests/` | compiler/model unit tests, mocked runtime tests, C simulation, hardware functional suite |
| `bench/` | FPGA, CPU and GPU benchmark drivers, power sampling, latency distribution, campaign scripts |
| `circuits/` | benchmark circuit generators (fixed seeds) and exported OpenQASM |
| `results/raw/` | every raw measurement (JSON/JSONL, power CSV, logs); `results/raw/env/` holds captured versions |
| `results/processed/` | tables derived from the raw data |
| `analysis/` | `load.py` (all runs in one table), `make_tables.py`, `make_figures.py` |
| `paper/generated/` | LaTeX macros and tables generated from the raw data |
| `paper/manuscript/` | the manuscript (single LaTeX source with figures, and the compiled PDF) |
| `docs/` | architecture, code map, baseline audit, provenance |

Bitstreams (`sv.xclbin`, about 65 to 80 MB each) are not stored in this repository. Rebuild them with the commands
below (section "Rebuilding the bitstreams"); `builds/<name>/sv.xclbin.sha256` holds the hash of the bitstream used for
the published measurements, and `builds/<name>/sv.xclbin.info` its metadata. Of the Vivado reports, the utilisation
reports are kept in full and the timing report as a summary extract (`timing_summary_extract.txt`).

| Build | Engines | Notes |
|---|---|---|
| `sv_e16w_m4` | 16 | main build: 256-bit ports, up to 4 high qubits per pass |
| `sv_e16w512` | 16 | 512-bit ports |
| `sv_e16w`, `sv_e16` | 16 | 256-bit ports, 2 high qubits per pass |
| `sv_e8` | 8 | 8-engine variant |

## Requirements

- FPGA: Alveo U55C, shell `xilinx_u55c_gen3x16_xdma_base_3` (platform `xilinx_u55c_gen3x16_xdma_3_202210_1`), XRT 2.16.
  Vitis 2023.2 only to rebuild bitstreams.
- Host: Python 3.8 with `pyxrt`, NumPy 1.24, Qiskit 1.2.4.
- CPU baselines: Python 3.11, Qiskit Aer 0.17.2, qsimcirq 0.22.1, Qulacs 0.6.14.
- GPU baselines: NVIDIA A100 SXM4 80 GB (Qiskit Aer GPU with and without cuStateVec, CUDA-Q), driven by
  `bench/gpu/campaign_a100.sh` through a batch scheduler (`run_remote.sh`, `queue_micro.sh`), and a local GPU host
  through `bench/gpu/run_local_gpu.sh`. Raw records are in `results/raw/gpu/` (`superseded/` holds earlier
  repetitions that the analysis does not use).

Exact versions used for the published measurements are in `results/raw/env/`.

## Quick start

All scripts expect to be run from the repository root after `source env.sh`, which keeps environments and caches
inside the repository (`envs/`, `cache/`, `tmp/`).

```
source env.sh
envs/fpga/bin/python tests/test_compiler_model.py      # compiler vs golden model (no FPGA needed)
envs/fpga/bin/python tests/test_runtime_mock.py        # runtime with mocked XRT (no FPGA needed)
envs/fpga/bin/python tests/run_suite_sv.py <BDF> builds/sv_e16w_m4/sv.xclbin 16 4 out.json 24,26,28,30
```

`<BDF>` is the card's PCIe address as shown by `xbutil examine`.

## Rebuilding the bitstreams

```
v++ -c -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k sv_engine --hls.clock 300000000:sv_engine -o sv_engine.xo kernels/sv/sv_engine.cpp
v++ -c -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k sv_swap   --hls.clock 300000000:sv_swap   -o sv_swap.xo   kernels/sv/sv_swap.cpp
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config builds/<name>/link.cfg --kernel_frequency 300 -o sv.xclbin sv_engine.xo sv_swap.xo
```

`builds/link_sv.sh <name> <engines> [pc_stride] [swap_CUs] [pcs_per_engine]` writes a `link.cfg` and links.
Add `-D SV_WIDE` to the engine compile for the 512-bit variant. `logs/build_portlimit_check.log` shows the
HBM port-connection limit (32 engines + 1 swap CU fails to link with 33 connections).

## Workflow: from a hardware run to the paper

```
kernels/ + host/          HLS engine and host compiler, tested without hardware (tests/)
      |  v++ compile and link (builds/<name>/link.cfg)
      v
builds/<name>/            bitstream hash, metadata and post route reports
      |  bench/*/campaign*.sh on the cards, the CPU and the GPU
      v
results/raw/              one JSON or JSONL record per run, with power and temperature logs
      |  analysis/make_tables.py, analysis/make_figures.py
      v
paper/generated/          every table and number of the paper, as LaTeX
      |
      v
paper/manuscript/         manuscript.tex and manuscript.pdf
```

No number in the manuscript is typed by hand: each one is a macro or a table in `paper/generated/`, produced by the
analysis scripts from `results/raw/`.

## Reproducing the measurements

```
FLAG=sv_e16w_m4 bench/fpga/campaign.sh <BDF> main|ablate|energy|builds|jitter|suite
bench/cpu/campaign.sh; bench/cpu/campaign_extra.sh      # CPU baselines, host otherwise idle
bench/gpu/run_local_gpu.sh <tag>                        # GPU baselines on a local GPU
bench/gpu/campaign_a100.sh                              # GPU baselines on the A100 node
envs/cpu/bin/python analysis/make_tables.py && envs/cpu/bin/python analysis/make_figures.py
```

FPGA runs pace themselves thermally: each repetition starts below 80 C, and a run aborts at 90 C (FPGA) or 85 C (HBM).
The analysis scripts read only `results/raw/`.

Host names in raw logs and reports have been replaced by placeholders (`fpga-host`, `<WS>` for the repository path).

## Citation

A. Tiwari, N. Ali, N. Khan, I. Mazhar, K. Islur, H. Gupta, S. Kumar, P. Tyagi, "FPGA Based Quantum Circuit Simulator:
Processing 30 Qubit Systems", Research Square preprint (2026), doi:10.21203/rs.3.rs-8581344/v1 (under major revision,
The Journal of Supercomputing).

## Author

Nasir Ali, Centre for Development of Advanced Computing (C-DAC), Noida, India.

## License

MIT, see `LICENSE`.
