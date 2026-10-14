// importer.js
// Turns files the user picked or dropped into a ready-to-place three.js
// object, or a clear error message. Never throws anything a user cannot read.
//
// Supported: .glb, .gltf (with its .bin and texture files dropped together),
// .obj (with an optional .mtl). Everything is read in memory from the File
// objects, so it works the same on macOS, Windows and Linux, in the desktop
// app and in a browser.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

export const MODEL_EXTENSIONS = ["glb", "gltf", "obj"];
export const ACCEPT = ".glb,.gltf,.obj,.bin,.mtl,.png,.jpg,.jpeg,.webp,.ktx2";
export const MAX_BYTES = 300 * 1024 * 1024; // above this, refuse: the app would stall
export const WARN_BYTES = 80 * 1024 * 1024; // above this, load but warn it may be slow

export class ImportError extends Error {}

const ext = (name) => name.split(".").pop().toLowerCase();
const baseName = (path) => decodeURIComponent(path.split(/[\\/]/).pop().split("?")[0]);
export const formatBytes = (n) =>
  n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;

// Picks the model file out of everything that was dropped.
export function chooseMainFile(files) {
  const models = files.filter((f) => MODEL_EXTENSIONS.includes(ext(f.name)));
  if (!models.length) {
    throw new ImportError("Berkelium can import .glb, .gltf and .obj files. Drop one of those.");
  }
  // Prefer the self-contained format if several were dropped together.
  return models.find((f) => ext(f.name) === "glb") || models.find((f) => ext(f.name) === "gltf") || models[0];
}

function describe(object) {
  let triangles = 0;
  let meshes = 0;
  const materials = new Set();
  const textures = new Set();
  object.traverse((o) => {
    if (!o.isMesh) return;
    meshes += 1;
    const g = o.geometry;
    triangles += g.index ? g.index.count / 3 : (g.attributes.position?.count || 0) / 3;
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (!m) continue;
      materials.add(m);
      for (const v of Object.values(m)) if (v && v.isTexture) textures.add(v);
    }
  });
  return { triangles: Math.round(triangles), meshes, materials: materials.size, textures: textures.size };
}

// Imported files can come in any unit (millimetres, kilometres). Very large or
// tiny models are scaled to a viewable size; normal ones keep their real size.
export function placement(object) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty() || !Number.isFinite(box.min.x) || !Number.isFinite(box.max.x)) {
    throw new ImportError("This file has no visible 3D geometry.");
  }
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  let scale = 1;
  let note = null;
  if (maxDim > 50) {
    scale = 4 / maxDim;
    note = "Very large model, scaled down to fit the view.";
  } else if (maxDim < 0.02) {
    scale = 1.5 / maxDim;
    note = "Very small model, scaled up so you can see it.";
  }
  const center = box.getCenter(new THREE.Vector3());
  // Sit the model on the floor, centred on the grid.
  const p = [-center.x * scale, -box.min.y * scale, -center.z * scale];
  return { transform: { p, r: [0, 0, 0], s: [scale, scale, scale] }, note };
}

function friendlyLoaderError(err, mainName) {
  const msg = String(err?.message || err || "");
  if (/draco/i.test(msg)) return `${mainName} uses Draco compression, which Berkelium does not support yet. Export it without Draco.`;
  if (/ktx2|basis/i.test(msg)) return `${mainName} uses KTX2 textures, which Berkelium does not support yet.`;
  return `Could not read ${mainName}. The file may be damaged or not a real 3D file.`;
}

export async function loadModelFiles(fileList) {
  const files = [...fileList];
  if (!files.length) throw new ImportError("No file was selected.");
  const main = chooseMainFile(files);
  const total = files.reduce((n, f) => n + f.size, 0);
  if (total > MAX_BYTES) {
    throw new ImportError(`${main.name} is ${formatBytes(total)}. The limit is ${formatBytes(MAX_BYTES)}.`);
  }

  // Companion files (.bin, textures, .mtl) are looked up by file name, so a
  // .gltf that references "textures/wood.png" finds a dropped "wood.png".
  const byName = new Map(files.map((f) => [f.name.toLowerCase(), f]));
  const blobUrls = new Map();
  const missing = new Set();
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => {
    if (/^(data|blob):/.test(url)) return url;
    const name = baseName(url).toLowerCase();
    const file = byName.get(name);
    if (!file) {
      missing.add(baseName(url));
      return url;
    }
    if (!blobUrls.has(name)) blobUrls.set(name, URL.createObjectURL(file));
    return blobUrls.get(name);
  });

  const kind = ext(main.name);
  let object;
  const warnings = [];
  try {
    if (kind === "glb" || kind === "gltf") {
      const loader = new GLTFLoader(manager).setMeshoptDecoder(MeshoptDecoder);
      const data = kind === "glb" ? await main.arrayBuffer() : await main.text();
      const gltf = await new Promise((resolve, reject) => loader.parse(data, "", resolve, reject));
      object = gltf.scene || gltf.scenes?.[0];
    } else {
      const loader = new OBJLoader(manager);
      const mtl = files.find((f) => ext(f.name) === "mtl");
      if (mtl) {
        const materials = new MTLLoader(manager).parse(await mtl.text(), "");
        materials.preload();
        loader.setMaterials(materials);
      }
      object = loader.parse(await main.text());
      if (!mtl) {
        // OBJ without materials renders as flat white. Give it a neutral
        // studio material so its shape reads clearly.
        const mat = new THREE.MeshStandardMaterial({ color: 0xc8ccd4, roughness: 0.55, metalness: 0.1 });
        object.traverse((o) => { if (o.isMesh) o.material = mat; });
      }
    }
  } catch (err) {
    if (missing.size) {
      throw new ImportError(`${main.name} needs ${[...missing].slice(0, 3).join(", ")} too. Select or drop them together.`);
    }
    throw new ImportError(friendlyLoaderError(err, main.name));
  } finally {
    // Textures are decoded by now; give the memory back.
    setTimeout(() => blobUrls.forEach((u) => URL.revokeObjectURL(u)), 5000);
  }

  if (!object) throw new ImportError(`${main.name} contains no scene.`);
  const stats = describe(object);
  if (!stats.meshes) throw new ImportError(`${main.name} has no 3D geometry to show.`);
  if (missing.size) warnings.push(`Missing ${[...missing].slice(0, 3).join(", ")}. Some textures may not show.`);
  if (total > WARN_BYTES) warnings.push(`Large file (${formatBytes(total)}). The viewport may be slower.`);

  object.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
      o.userData.pickable = true;
    }
  });

  const name = main.name.replace(/\.[^.]+$/, "") || "Model";
  return { object, name, fileName: main.name, bytes: total, stats, warnings };
}
