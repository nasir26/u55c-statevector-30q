# Source in every shell from the repository root:  source env.sh
export WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR=$WS/tmp
export XDG_CACHE_HOME=$WS/cache
export PIP_CACHE_DIR=$WS/cache/pip
export CONDA_PKGS_DIRS=$WS/cache/conda_pkgs
export CONDA_ENVS_PATH=$WS/envs
export NUMBA_CACHE_DIR=$WS/cache/numba
export CUPY_CACHE_DIR=$WS/cache/cupy
export MPLCONFIGDIR=$WS/cache/matplotlib
export PYTHONNOUSERSITE=1          # ~/.local site-packages otherwise leak into envs
export PYTHONDONTWRITEBYTECODE=1
export XILINX_LOCAL_USER_DATA=no   # stop Vitis tools writing into ~/.Xilinx
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
export PYTHONPATH=/opt/xilinx/xrt/python${PYTHONPATH:+:$PYTHONPATH}
vitis_env() { source /tools/Xilinx/Vitis/2023.2/settings64.sh; }
export WGETRC=$WS/.wgetrc   # keep wget from writing ~/.wget-hsts
export MAMBA_ROOT_PREFIX=$WS/envs/mamba
export PATH=$WS/envs/bin:$PATH
mm() { CONDA_ENVS_PATH= HOME=$WS/tmp/fakehome micromamba --no-rc -r $MAMBA_ROOT_PREFIX "$@"; }   # fake HOME stops ~/.conda/environments.txt registration
export PATH=$WS/envs/texlive/bin/x86_64-linux:$PATH   # TeX Live (workspace)
export PATH=$WS/envs/numactl/bin:$PATH   # numactl built from source (v2.0.18)
