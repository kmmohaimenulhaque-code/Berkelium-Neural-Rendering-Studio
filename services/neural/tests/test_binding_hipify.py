"""Regression guard for the ROCm PyTorch binding (csrc/torch_bindings.cpp), which the CPU kernel tests never compile.

Runs PyTorch's own hipify on a temporary copy of csrc/ (exactly what torch.utils.cpp_extension does on ROCm) and
syntax-checks the translated binding against the installed torch headers, with two tiny stand-in headers for the
ROCm-only includes. A syntax check is NOT a ROCm build; the real build is exercised by tests marked ``hip``.
"""

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parents[1] / "csrc"
SHIM_HIP = """#pragma once
typedef enum hipError_t { hipSuccess = 0, hipErrorInvalidValue = 1 } hipError_t;
typedef struct ihipStream_t* hipStream_t;
inline const char* hipGetErrorString(hipError_t) { return ""; }
"""
SHIM_CTX = """#pragma once
#include <hip/hip_runtime.h>
namespace at { namespace cuda { struct ShimStream { operator hipStream_t() const { return nullptr; } };
inline ShimStream getCurrentCUDAStream() { return {}; } } }
"""


def test_rocm_binding_hipifies_and_parses(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.utils.cpp_extension import include_paths
    from torch.utils.hipify import hipify_python
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    src = tmp_path / "src"
    shutil.copytree(CSRC, src)
    hipify_python.hipify(project_directory=str(src), output_directory=str(src), includes=[str(src / "*")],
                         show_detailed=False, is_pytorch_extension=True)
    out = [p for p in src.iterdir() if p.name.startswith("torch_bindings") and p.suffix == ".cpp"]
    translated = next((p for p in out if "hip" in p.name), src / "torch_bindings.cpp")
    shim = tmp_path / "shim"
    (shim / "hip").mkdir(parents=True)
    (shim / "ATen" / "hip").mkdir(parents=True)
    (shim / "hip" / "hip_runtime.h").write_text(SHIM_HIP)
    (shim / "ATen" / "hip" / "HIPContext.h").write_text(SHIM_CTX)
    (shim / "ATen" / "cuda").mkdir(parents=True)
    (shim / "ATen" / "cuda" / "CUDAContext.h").write_text(SHIM_CTX)
    cmd = ["g++", "-std=c++20", "-fsyntax-only", f"-I{shim}", f"-I{src}", *[f"-I{p}" for p in include_paths()],
           f"-I{sysconfig.get_paths()['include']}", "-DTORCH_EXTENSION_NAME=berkelium_det_bwd", str(translated)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    assert torch.__version__
