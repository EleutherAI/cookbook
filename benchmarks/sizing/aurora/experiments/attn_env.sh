# Sourced by the attention payloads: keep Triton and driver caches inside $RUN.
export PYTHONDONTWRITEBYTECODE=1 TRITON_HOME=$RUN/triton_home TRITON_CACHE_DIR=$RUN/triton_home/cache
export XDG_CACHE_HOME=$RUN/xdg_cache NEO_CACHE_DIR=$RUN/neo_cache
unset TRITON_AUTOTUNE_CACHE_DIR
mkdir -p $TRITON_CACHE_DIR $XDG_CACHE_HOME $NEO_CACHE_DIR
