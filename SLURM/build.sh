#!/bin/bash
# Builds SLURM from source into ~/.local/slurm, without root. Each machine builds its own copy
# (paths under $HOME are compiled in, and homes differ between machines), so builds only need to
# interoperate: same SLURM version everywhere (pinned here, tarball checksummed) with the same
# patches (patches/*.patch), plus the munge key and config, which bin/setup_node handles. Other
# packages are unpinned.
#
# Usage: build.sh            build
#        build.sh --version  print what it builds: the SLURM version and a hash of the patches,
#                            eg. 25.11.8+3f2a9c1e (setup_node rebuilds if the installed build,
#                            recorded in ~/.local/slurm/.slurm_build, differs)
# SLURM_PREFIX / SLURM_BUILD_DIR override ~/.local/slurm and ~/.local/slurm-build (for testing).
set -euo pipefail
version=25.11.8
tarball_sha256=34ace13f81011add6094569d13bfc4006ad8868201c2236e2905443c7e526393
runtime_pkgs=(munge json-c lz4-c readline)
# conda-forge's default sysroot is the newest glibc (2.39), whose binaries don't run on older
# systems such as Ubuntu 22.04 (2.35); 2.28 runs everywhere.
build_pkgs=(c-compiler cxx-compiler make pkg-config 'sysroot_linux-64=2.28')
prefix=${SLURM_PREFIX:-$HOME/.local/slurm}        # runtime libs + SLURM itself (+ etc/, var/)
build=${SLURM_BUILD_DIR:-$HOME/.local/slurm-build} # compilers + source

patches=("$(cd "$(dirname "$0")" && pwd)"/patches/*.patch)
# The hash covers only the patches' code changes (from their first "diff"), so editing their notes
# doesn't make every machine rebuild.
build_id=$version+$(for p in "${patches[@]}"; do sed -n '/^diff /,$p' "$p"; done | sha256sum | cut -c1-8)
if [ "${1:-}" = --version ]; then echo "$build_id"; exit 0; fi

conda=${CONDA_EXE:-}
[ -n "$conda" ] || conda=$(command -v conda || true)
for c in "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [ -n "$conda" ] || { [ -x "$c" ] && conda=$c; } || true
done
[ -n "$conda" ] || { echo "build.sh: conda not found (set CONDA_EXE)" >&2; exit 1; }
for env in "$prefix" "$build"; do
    pkgs=("${runtime_pkgs[@]}"); [ "$env" = "$build" ] && pkgs=("${build_pkgs[@]}")
    [ -d "$env/conda-meta" ] || "$conda" create -y -q -p "$env" -c conda-forge --override-channels "${pkgs[@]}" > /dev/null
done

mkdir -p "$build/src" && cd "$build/src"
[ -f "slurm-$version.tar.bz2" ] || curl -sSfLO "https://download.schedmd.com/slurm/slurm-$version.tar.bz2"
echo "$tarball_sha256  slurm-$version.tar.bz2" | sha256sum -c --quiet
rm -rf "slurm-$version" && tar xjf "slurm-$version.tar.bz2" && cd "slurm-$version"
for p in "${patches[@]}"; do patch -p1 -s < "$p"; done
export PATH=$build/bin:$PATH
echo "Building SLURM $version (logs in $build/src/)"
# Without CC set explicitly, configure picks the system gcc (and the system glibc).
CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ \
    PKG_CONFIG_PATH=$prefix/lib/pkgconfig CPPFLAGS="-I$prefix/include" \
    LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib" \
    ./configure --prefix="$prefix" --sysconfdir="$prefix/etc" \
    --with-munge="$prefix" --with-json="$prefix" --with-lz4="$prefix" \
    --without-hwloc --without-pmix --without-ucx --without-nvml --without-rsmi --without-hdf5 \
    --without-rdkafka --without-libcurl --disable-pam > ../configure.log 2>&1
make -j"$(nproc)" > ../make.log 2>&1
make install > ../install.log 2>&1

# The binaries must run on this machine's glibc.
need=$(for f in "$prefix"/sbin/slurm* "$prefix"/bin/s* "$prefix"/lib/slurm/*.so; do
    { objdump -T "$f" 2>/dev/null || true; } | grep -oE 'GLIBC_[0-9.]+' || true; done | sort -uV | tail -1)
have=GLIBC_$(getconf GNU_LIBC_VERSION | cut -d' ' -f2)
if [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | tail -1)" != "$have" ]; then
    echo "build.sh: binaries need $need, but this machine has $have" >&2
    exit 1
fi
echo "$build_id" > "$prefix/.slurm_build"
echo "Installed SLURM $build_id; needs $need, machine has $have"
