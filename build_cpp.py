"""
Build the pyMoM3d C++ extension in-place.

Usage
-----
    venv/bin/python build_cpp.py build_ext --inplace

This compiles ``src/cpp/mom_kernel.cpp`` (pybind11 + OpenMP) and places the
resulting shared library at::

    src/pyMoM3d/mom/_cpp_kernels.<platform-tag>.so

The extension is then importable as ``pyMoM3d.mom._cpp_kernels``.

Strata backend (optional)
--------------------------
To also build the Strata multilayer Green's function wrapper, first install
the Strata C++ library (https://github.com/modelics/strata), then pass its
installation prefix::

    venv/bin/python build_cpp.py build_ext --inplace --with-strata=/usr/local

This compiles ``src/cpp/strata_kernels.cpp`` against the installed ``libstrata``
and places the result at::

    src/pyMoM3d/greens/layered/strata_kernels.<platform-tag>.so

OpenMP on macOS
---------------
Apple clang does not support ``-fopenmp`` natively.  libomp from Homebrew
provides the runtime.  This script detects its location automatically via
``brew --prefix libomp`` and adds the required flags.  If libomp is not found,
the extension is built single-threaded (still correct, just slower).

CMake alternative
-----------------
A ``src/cpp/CMakeLists.txt`` is also provided for users who prefer cmake:

    cmake -S src/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
    cmake --build build/cpp -- -j$(nproc)
"""

import argparse
import subprocess
import sys
import os
import platform
import numpy as np
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# ---------------------------------------------------------------------------
# Parse --with-strata before setuptools sees argv
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    '--with-strata', metavar='PATH', default=None,
    help='Path to Strata installation prefix (enables strata_kernels build)'
)
_known, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining
strata_dir = _known.with_strata

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
compile_args = ["-O3", "-march=native", "-std=c++17", "-DNDEBUG"]
link_args: list[str] = []
include_dirs = [np.get_include(), "src/cpp"]
define_macros: list[tuple] = []

# ---------------------------------------------------------------------------
# OpenMP detection
# ---------------------------------------------------------------------------
def _brew_prefix(pkg: str) -> str | None:
    """Return the Homebrew prefix for a package, or None."""
    try:
        result = subprocess.check_output(
            ["brew", "--prefix", pkg],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return result if result else None
    except Exception:
        return None


def _try_omp_flags(cc: list[str], extra_compile: list[str], extra_link: list[str]) -> bool:
    """Return True if the compiler accepts the given OMP flags."""
    import tempfile
    src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        src_path = f.name
    out_path = src_path.replace(".cpp", ".out")
    try:
        cmd = cc + extra_compile + [src_path, "-o", out_path] + extra_link
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        for p in (src_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


cc = [os.environ.get("CXX", "c++")]
omp_compile: list[str] = []
omp_link: list[str] = []
omp_enabled = False

if platform.system() == "Darwin":
    # Apple clang: needs -Xpreprocessor -fopenmp and libomp from Homebrew
    prefix = _brew_prefix("libomp")
    if prefix:
        candidate_compile = ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"]
        candidate_link = [f"-L{prefix}/lib", "-lomp"]
        if _try_omp_flags(cc, candidate_compile, candidate_link):
            omp_compile = candidate_compile
            omp_link = candidate_link
            omp_enabled = True
else:
    # Linux / other: standard -fopenmp
    if _try_omp_flags(cc, ["-fopenmp"], ["-fopenmp"]):
        omp_compile = ["-fopenmp"]
        omp_link = ["-fopenmp"]
        omp_enabled = True

if omp_enabled:
    print("[build_cpp] OpenMP enabled")
    compile_args.extend(omp_compile)
    link_args.extend(omp_link)
else:
    print("[build_cpp] OpenMP NOT available — building single-threaded")

# ---------------------------------------------------------------------------
# Extension: MoM kernel (always built)
# ---------------------------------------------------------------------------
ext_mom = Pybind11Extension(
    "pyMoM3d.mom._cpp_kernels",
    sources=["src/cpp/mom_kernel.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=define_macros,
    cxx_std=17,
)

ext_modules = [ext_mom]

# ---------------------------------------------------------------------------
# Extension: Strata GF wrapper (optional — requires --with-strata=PATH)
# ---------------------------------------------------------------------------
if strata_dir is not None:
    # Strata installs headers to <prefix>/inc and library to <prefix>/lib
    strata_inc = os.path.join(strata_dir, 'inc')
    strata_lib_dir = os.path.join(strata_dir, 'lib')

    if not os.path.isdir(strata_inc):
        print(f"[build_cpp] WARNING: Strata include dir not found: {strata_inc}")
    if not os.path.isdir(strata_lib_dir):
        print(f"[build_cpp] WARNING: Strata lib dir not found: {strata_lib_dir}")

    strata_compile = list(compile_args)          # inherit -O3, -march=native, OMP
    strata_link    = list(link_args) + [f"-Wl,-rpath,{strata_lib_dir}"]

    ext_strata = Pybind11Extension(
        "pyMoM3d.greens.layered.strata_kernels",
        sources=["src/cpp/strata_kernels.cpp"],
        include_dirs=[np.get_include(), strata_inc, "src/cpp"],
        library_dirs=[strata_lib_dir],
        libraries=["strata"],
        extra_compile_args=strata_compile,
        extra_link_args=strata_link,
        cxx_std=17,
    )
    ext_modules.append(ext_strata)
    print(f"[build_cpp] Strata backend enabled (prefix: {strata_dir})")
else:
    print("[build_cpp] Strata backend NOT built  "
          "(pass --with-strata=/path/to/strata to enable)")

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
setup(
    name="pyMoM3d",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
