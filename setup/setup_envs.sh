#!/bin/bash
# Creates the two Python environments of the package with the bundled micromamba (no system change, no root):
#   envs/fpga : Python 3.8, Qiskit 1.2.4, Qiskit Aer 0.15.1, NumPy 1.24.4 (pyxrt of XRT is used from /opt/xilinx/xrt/python)
#   envs/cpu  : Python 3.11, analysis and figure scripts
# and unpacks the bundled TeX Live (envs/texlive) for the document builds.
# Author: Nasir Ali, C-DAC Noida.
set -eu
P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source $P/env.sh
MM=${MICROMAMBA:-$P/setup/micromamba}; [ -x $MM ] || MM=$(command -v micromamba)
mkdir -p $P/envs
for e in fpga cpu; do
  [ -x $P/envs/$e/bin/python ] && { echo "envs/$e exists"; continue; }
  $MM create -y -p $P/envs/$e -f $P/setup/env_$e.yml
done
if [ ! -d $P/envs/texlive ] && [ -f $P/setup/texlive.tar.gz ]; then tar -C $P/envs -xzf $P/setup/texlive.tar.gz; fi
$P/envs/fpga/bin/python -c "import qiskit, qiskit_aer, numpy; print('fpga env', qiskit.__version__, qiskit_aer.__version__, numpy.__version__)"
$P/envs/fpga/bin/python -c "import uuid, pyxrt; print('pyxrt ok')" || echo "pyxrt import failed: XRT python bindings must match Python 3.8 (see README)"
$P/envs/cpu/bin/python -c "import numpy, matplotlib; print('cpu env ok')"
[ -x $P/envs/texlive/bin/x86_64-linux/pdflatex ] && $P/envs/texlive/bin/x86_64-linux/pdflatex --version | head -1
