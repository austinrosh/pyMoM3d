"""
Build the pyMoM3d C++ extension in-place.

Usage
-----
    venv/bin/python build_cpp.py build_ext --inplace

This compiles ``src/cpp/mom_kernel.cpp`` (pybind11 + OpenMP) and places the
resulting shared library at::

    src/pyMoM3d/mom/_cpp_kernels.<platform-tag>.so

The extension is then importable as ``pyMoM3d.mom._cpp_kernels``.

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

import subprocess
import sys
import os
import platform
import numpy as np
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
# Extension definition
# ---------------------------------------------------------------------------
ext = Pybind11Extension(
    "pyMoM3d.mom._cpp_kernels",
    sources=["src/cpp/mom_kernel.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=define_macros,
    cxx_std=17,
)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
setup(
    name="pyMoM3d",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
