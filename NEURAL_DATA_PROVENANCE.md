# Neural Data and Artifact Provenance

**Revision:** `NR-0.2` (2026-10-12)  
**Status:** proposed manifest and review ledger; no models, datasets, or weights were downloaded for this documentation task.

## Rules

1. Every input, generated file, model, source checkout, dependency, and report receives a SHA-256 and an opaque artifact ID.
2. A URL, family name, release label, or branch is not a reproducibility pin. Record commit/snapshot, file list, byte size, and retrieval date.
3. Code license, model-weight license, base-model license, dataset/image rights, and dependency notices are separate fields. A permissive code license does not authorize a checkpoint or output use.
4. Unknown scale is `normalized_object`, never meters. Camera basis, pose direction, intrinsics, crop/resize, and normalization transforms are mandatory for geometric artifacts.
5. Provenance is append-only. A derived artifact references parents and the exact tool/config revision; it never overwrites a source receipt.
6. A `planned`, `blocked`, or `failed` receipt is not a quality result. A completed job is not an experiment pass without validation evidence.

## Minimum manifest

```json
{
  "artifactId": "opaque-id",
  "kind": "image | camera-set | gaussian-scene | triangle-mesh | cad-model | validation-report | checkpoint | experiment-receipt",
  "schemaVersion": "1",
  "sha256": "hex",
  "byteSize": 0,
  "mediaType": "application/octet-stream",
  "parentArtifactIds": [],
  "source": {"url": "", "revision": "", "retrievedAt": ""},
  "license": {"code": "", "weights": "", "data": "", "notices": [], "scope": ""},
  "tool": {"name": "", "revision": "", "configSha256": ""},
  "camera": {"convention": "", "poseDirection": "", "intrinsicsUnits": "pixels"},
  "scene": {"units": "normalized_object", "worldBasis": "", "normalization": ""},
  "validation": {"status": "not_run", "reportArtifactId": null},
  "resourceReceipt": {"provider": "", "gpuCount": 0, "billedUnits": 0},
  "warnings": []
}
```

For binary arrays also record dtype, byte order, shape, stride, padded capacity, and alignment. For images record color space, alpha semantics, dimensions, mask method, and crop/pad transform. For Gaussian scenes record the renderer schema, parameterization, active count, ID policy, and camera manifest. For mesh/CAD record topology/material format, units, basis transform, and whether watertightness or dimensional accuracy was actually measured.

## Reviewed source ledger

| Source | Pinned identity | Intended use | Permission/status |
| --- | --- | --- | --- |
| Berkelium target archive | SHA-256 `f1a6a8e52abbbbb2c93a24a85d80b40cead9c536a458bc5630f3bbed1d7958f4`; no `.git` metadata | Documentation/integration target | Local repository MIT applies to independently authored files; archive ancestry is unknown. |
| Engineering reference | `737dc35c42aaf97f52a6ac984a806adef9518525`, `origin/physics/p1-numerical`; remote `main` `70cc43ee41b8dec2de515be35d27e98b952337b` | API/schema/geometry contract reference | README license remains unresolved; no source transfer. |
| AMD GSplat | release `1.5.3b2`, revision `ff8bc5489545bcc1b5a2b997d975834429a0d7af` | Comparison/reference and possible separately reviewed backend | NOTICE indicates Apache-2.0 upstream portions and MIT AMD modifications; retain notices. |
| Z-Image code/model | code `26f23eda626ffadda020b04ff79488e1d72004cd`; Turbo snapshot `f332072aa78be7aecdf3ee76d5c247082da564a6` | Optional image prior | Record code, checkpoint, tokenizer, VAE, and dependency terms independently; snapshot files are approximately 32.83 GB. |
| TripoSR | code `107cefdc244c39106fa830359024f6a2f1c78871`; model snapshot `5b521936b01fbe1890f6f9baed0254ab6351c04a` | First explicit 3D prior candidate | Upstream declares MIT. **Used in M6 (2026-10-15)**: `model.ckpt` sha256 `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee`; torchmcubes replaced by CPU marching cubes (D-020); only Berkelium-owned control images were input. |
| Google Scanned Objects (Google LLC) | Fuel v1 zips: `Schleich_Lion_Action_Figure` `ae402d06…22ad9`, `Racoon` `1408b68f…f5dd7`, `Room_Essentials_Bowl_Turquiose` `6eb64260…c4b9` (full hashes in `fitting/real_object.py`) | M7 real-object observations + evaluation | CC-BY 4.0 with attribution (THIRD_PARTY_NOTICES.md). Scans are fetched, not redistributed; observations are renders of the scans, labelled as such. |
| Zero123++ / InstantMesh | Zero123++ code `7d0315c31be6eb906b34cf07d91310f8e12e9b95`; InstantMesh code `08822c52fdc399b93ea00e4fa9e596344ed52ccc` | Non-commercial comparison only | Zero123++ weights are CC-BY-NC-4.0; not a production dependency without separate permission review. |
| MultiNeRF/Mip-NeRF 360 | `5b4d4f64608ec8077222c52fdf814d40acc10bc1` | Later real-capture comparison | Code/data/image rights remain separate; no dataset download assumed. |

## Required review before promotion

- Reconcile every dependency and checkpoint license with the intended distribution, including transitive native extensions.
- Verify the artifact's parent chain, hashes, camera transforms, units, and one declared alignment transform.
- Attach measured validation and resource receipts; never substitute upstream benchmark numbers for Berkelium measurements.
- Preserve failed/blocked examples and warnings. Do not delete a bad generated image, mesh, or view set merely because it hurts aggregate quality.
