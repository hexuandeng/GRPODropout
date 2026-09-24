TMP_ROOT=${TMP_ROOT:-/root/yzh/verl_runtime}
mkdir -p ${TMP_ROOT}/{tmp,ray,triton,inductor,xdg}
chmod 1777 ${TMP_ROOT}/tmp || true

export TMPDIR=${TMP_ROOT}/tmp
export TMP=${TMP_ROOT}/tmp
export TEMP=${TMP_ROOT}/tmp
export RAY_TMPDIR=${TMP_ROOT}/ray
export RAY_DISABLE_DASHBOARD=${RAY_DISABLE_DASHBOARD:-1}
export TRITON_CACHE_DIR=${TMP_ROOT}/triton
export TORCHINDUCTOR_CACHE_DIR=${TMP_ROOT}/inductor
export XDG_CACHE_HOME=${TMP_ROOT}/xdg
