# E-T3 P0 state check (2026-10-15)

| Item | State |
|---|---|
| Branch / HEAD | `Berkelium-Neural-Rendering-Studio-neural-v1` @ `d2dcf2b` (HANDOFF.md), pushed earlier through `d2dcf2b` |
| `git stash list` | empty |
| Uncommitted user changes | none; untracked: `services/neural/berkelium_neural/text3d/shap_e_backend.py` (partial, from the interrupted session) and `shap_e_model_cache/` |
| `shap_e_model_cache/` | one file, `ViT-L-14.pt`, 932,768,134 B, sha256 `b8cca3fd…a03836` = pinned (complete; created by Shap-E's CLIP loader, whose cache defaults to `$CWD/shap_e_model_cache`). Moved to `data/models/shap_e/`; directory removed and git-ignored |
| `data/models/shap_e/` | partial `text_cond.pt` (167 MB of 1.26 GB) and `text_cond_config.yaml` from the interrupted run; resumed with `curl -C -` |
| `third_party/shap-e` | clone at `50131012ee11c9d2617f3886c10f000d3c7a3b43`, clean, installed editable (`--no-deps`) in `.venv-cpu` |
| Packages (`.venv-cpu`) | torch 2.14.1+cpu, torchvision 0.29.1+cpu, clip 1.0, ftfy, regex, blobfile, filelock, fire, humanize, pyyaml, trimesh, scikit-image present; ipywidgets absent (so `shap_e.util.notebooks` must never be imported); pytorch3d absent (STF falls back to Shap-E's native raycast renderer) |
| Partial backend review | rewritten: no explicit CLIP cache dir (CLIP would land in the CWD), TorchScript CLIP load, vision tower loaded then dropped (peak memory already paid), no stage progress/cancel, no `available()` weight check |
| Previous failed run | `shape_first.log`: CLIP download completed, then the process died right after `torch.jit.load` with no traceback (consistent with the memory watchdog below) |
| `.gitignore` | `/data/` already ignored (covers `data/models/`); added `/shap_e_model_cache/`. Committed weights: only the native `exports/*.pt` (362k-param model, adapter). No pretrained weights are committed |
| Host | 2 vCPU, 7.8 GB RAM (~3.9 GB available), no swap, no GPU; 8.5 GB free disk (not 12 GB) before downloads |
| Memory watchdog | no cgroup limit (`memory.max` = max); an external watchdog SIGTERMs processes at ~3.2 GB RSS. Reproduced twice (exit 143 at 3.24 GB and 3.26 GB max RSS, `/usr/bin/time -v`). RSS includes file-backed pages of memory-mapped weights |
| `/usr/bin/time` | absent; installed `time` 1.9 with apt for the P2 gate |
