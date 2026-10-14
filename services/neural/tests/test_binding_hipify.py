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
# Stand-in for the ROCm PyTorch stream API, mirroring the real shapes: c10::hip::HIPStreamMasqueradingAsCUDA (implicitly
# convertible to hipStream_t, as in ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h) exposed in at::hip, plus the
# at::cuda::getCurrentCUDAStream spelling that newer hipify versions leave untranslated. The same text backs both
# fake headers (ATen/hip/HIPContext.h and ATen/cuda/CUDAContext.h); one include guard keeps a translation unit that
# includes both free of duplicate definitions.
SHIM_CTX = """#pragma once
#ifndef BK_TEST_SHIM_ROCM_STREAMS
#define BK_TEST_SHIM_ROCM_STREAMS
#include <hip/hip_runtime.h>
namespace c10 { namespace hip {
class HIPStreamMasqueradingAsCUDA {
 public:
  hipStream_t stream() const { return nullptr; }
  operator hipStream_t() const { return stream(); }
};
inline HIPStreamMasqueradingAsCUDA getCurrentHIPStreamMasqueradingAsCUDA(signed char device_index = -1) {
  (void)device_index;
  return {};
}
} }
namespace at {
namespace hip {
using c10::hip::HIPStreamMasqueradingAsCUDA;
using c10::hip::getCurrentHIPStreamMasqueradingAsCUDA;
}
namespace cuda {
inline c10::hip::HIPStreamMasqueradingAsCUDA getCurrentCUDAStream(signed char device_index = -1) {
  return c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}
}
}
#endif
"""
# Stream spelling emitted by the hipify of PyTorch 2.8 (ROCm 7.0, MI300X run 2026-10-13) for
# at::cuda::getCurrentCUDAStream(); newer hipify versions (e.g. the CPU torch used in CI) leave the call unchanged.
ROCM_PT28_STREAM = ("at::cuda::getCurrentCUDAStream()", "at::hip::getCurrentHIPStreamMasqueradingAsCUDA()")


def _shim_dir(tmp_path):
    shim = tmp_path / "shim"
    for sub in ("hip", "ATen/hip", "ATen/cuda"):
        (shim / sub).mkdir(parents=True, exist_ok=True)
    (shim / "hip" / "hip_runtime.h").write_text(SHIM_HIP)
    (shim / "ATen" / "hip" / "HIPContext.h").write_text(SHIM_CTX)
    (shim / "ATen" / "cuda" / "CUDAContext.h").write_text(SHIM_CTX)
    return shim


def _syntax_check(shim, src, path, torch_includes=True):
    from torch.utils.cpp_extension import include_paths
    inc = [*[f"-I{p}" for p in include_paths()], f"-I{sysconfig.get_paths()['include']}"] if torch_includes else []
    cmd = ["g++", "-std=c++20", "-fsyntax-only", f"-I{shim}", f"-I{src}", *inc,
           "-DTORCH_EXTENSION_NAME=berkelium_det_bwd", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_stream_shims_are_consistent_and_includable_together(tmp_path):
    """The fake headers must expose both stream spellings with a stream convertible to hipStream_t, and must be
    includable together (the hipified binding plus anything else that pulls in the CUDA spelling)."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    shim = _shim_dir(tmp_path)
    probe = tmp_path / "probe.cpp"
    probe.write_text("""#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPContext.h>
#include <type_traits>
static hipError_t takes_stream(hipStream_t) { return hipSuccess; }
int main() {
  hipStream_t a = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  hipStream_t b = at::cuda::getCurrentCUDAStream();
  static_assert(std::is_convertible_v<at::hip::HIPStreamMasqueradingAsCUDA, hipStream_t>);
  return takes_stream(at::hip::getCurrentHIPStreamMasqueradingAsCUDA()) + takes_stream(a) + takes_stream(b);
}
""")
    r = _syntax_check(shim, tmp_path, probe, torch_includes=False)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("stream_spelling", ["installed_hipify", "rocm_pt28_hipify"])
def test_rocm_binding_hipifies_and_parses(tmp_path, stream_spelling):
    torch = pytest.importorskip("torch")
    from torch.utils.hipify import hipify_python
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    src = tmp_path / "src"
    shutil.copytree(CSRC, src)
    hipify_python.hipify(project_directory=str(src), output_directory=str(src), includes=[str(src / "*")],
                         show_detailed=False, is_pytorch_extension=True)
    out = [p for p in src.iterdir() if p.name.startswith("torch_bindings") and p.suffix == ".cpp"]
    translated = next((p for p in out if "hip" in p.name), src / "torch_bindings.cpp")
    text = translated.read_text()
    assert "#include <ATen/hip/HIPContext.h>" in text or "#include <ATen/cuda/CUDAContext.h>" in text
    if stream_spelling == "rocm_pt28_hipify" and ROCM_PT28_STREAM[0] in text:
        translated.write_text(text.replace(*ROCM_PT28_STREAM))   # what the PyTorch 2.8 ROCm hipify emits
    text = translated.read_text()
    assert ROCM_PT28_STREAM[0] in text or ROCM_PT28_STREAM[1] in text, "stream call not found in the binding"
    r = _syntax_check(_shim_dir(tmp_path), src, translated)
    assert r.returncode == 0, r.stderr[-3000:]
    assert torch.__version__
