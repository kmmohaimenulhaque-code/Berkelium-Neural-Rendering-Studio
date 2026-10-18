"""Berkelium neural rendering + text-to-3D service.

Layers (each behind a typed interface, see contracts.py):
  camera / scene        canonical OpenCV camera + GaussianScene (berkelium-gaussian-soa/1)
  render.reference      Berkelium CPU reference renderer (PyTorch, autograd, fp32/fp64)
  render.gsplat_adapter pinned AMD gsplat (ROCm) backend
  render.berkelium_ext  Berkelium-owned HIP extension (deterministic backward)
  text3d                owned procedural dataset, text-conditioned SDF generator, LoRA adapters
  service               immutable artifact store, async job manager, FastAPI router
"""

__version__ = "0.1.0"
