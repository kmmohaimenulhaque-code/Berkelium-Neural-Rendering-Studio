// shapes.js
// The basic shapes you can add from the toolbar or with Shift+A.
//
// Each shape is built centered on its own origin, like in Blender and Unity,
// so Rotate spins it around its middle and Scale grows it evenly. Placement
// lifts it by half its height so it rests on the floor instead of sinking
// halfway into it.
//
// Sizes are 1 unit across, a little smaller than the 1.2 unit starter cube,
// so a new shape next to it reads as a sibling, not a copy.

import * as THREE from "three";

const ACCENT = 0x00d4d4;
const EDGE_ANGLE = 30; // degrees. Only real corners get an outline, not the facets of a curved surface.

// halfHeight: distance from the origin down to the floor contact point.
// footprint: radius of the shape seen from above, used to avoid overlaps.
// edges: true when the shape has crisp corners an outline can follow.
// Curved shapes (sphere, torus) have none, so they get a bounding box instead.
export const SHAPES = [
  {
    kind: "cube",
    label: "Cube",
    halfHeight: 0.5,
    footprint: 0.71,
    edges: true,
    geometry: () => new THREE.BoxGeometry(1, 1, 1),
  },
  {
    kind: "sphere",
    label: "Sphere",
    halfHeight: 0.5,
    footprint: 0.5,
    edges: false,
    geometry: () => new THREE.SphereGeometry(0.5, 48, 32),
  },
  {
    kind: "cylinder",
    label: "Cylinder",
    halfHeight: 0.5,
    footprint: 0.5,
    edges: true,
    geometry: () => new THREE.CylinderGeometry(0.5, 0.5, 1, 48),
  },
  {
    kind: "cone",
    label: "Cone",
    halfHeight: 0.5,
    footprint: 0.5,
    edges: true,
    geometry: () => new THREE.ConeGeometry(0.5, 1, 48),
  },
  {
    kind: "plane",
    label: "Plane",
    // A plane lies on the floor. The small lift keeps it from flickering
    // against the shadow floor, which sits at exactly 0.
    halfHeight: 0.005,
    footprint: 1.42,
    edges: true,
    // Rotated into the geometry itself, so the node's rotation reads 0, 0, 0
    // in the Transform panel like every other shape.
    geometry: () => new THREE.PlaneGeometry(2, 2).rotateX(-Math.PI / 2),
  },
  {
    kind: "torus",
    label: "Torus",
    halfHeight: 0.55,
    footprint: 0.55,
    edges: false,
    geometry: () => new THREE.TorusGeometry(0.4, 0.15, 24, 64),
  },
];

export const shapeByKind = (kind) => SHAPES.find((s) => s.kind === kind) || null;

// Builds the three.js object for a shape. Every shape gets its own geometry
// and material, so freeing one (garbage collection after undo history moves
// on) can never break another.
export function makeShapeMesh(kind) {
  const def = shapeByKind(kind);
  if (!def) throw new Error(`unknown shape: ${kind}`);
  const geometry = def.geometry();
  const mesh = new THREE.Mesh(
    geometry,
    // Same studio look as the starter cube. envMapIntensity 0 keeps shapes
    // looking identical whether or not a model's reflection lighting is on.
    new THREE.MeshStandardMaterial({
      color: ACCENT,
      roughness: 0.35,
      metalness: 0.25,
      envMapIntensity: 0,
      side: kind === "plane" ? THREE.DoubleSide : THREE.FrontSide,
    })
  );
  mesh.name = def.label;
  mesh.castShadow = kind !== "plane";
  mesh.receiveShadow = true; // so a shape stacked on a plane or another shape shadows it
  mesh.userData.pickable = true;
  mesh.userData.shape = kind;

  if (def.edges) {
    const edges = new THREE.EdgesGeometry(geometry, EDGE_ANGLE);
    // Faint outline that is always on, matching the starter cube.
    mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })));
    // Bright outline shown only while selected. The engine toggles it.
    const selection = new THREE.LineSegments(edges.clone(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 }));
    selection.scale.setScalar(1.004); // a hair outside the faces so it never z-fights
    selection.visible = false;
    selection.raycast = () => {};
    mesh.add(selection);
    mesh.userData.selectionOutline = selection;
  }
  return mesh;
}

const FLOOR_LIMIT = 18; // keep new shapes on the 40 x 40 grid
const GAP = 0.3; // breathing room between shapes, so each one is easy to click

// Picks a floor spot [x, z] for a new shape. Tries the center first, then
// rings of spots around it, nearest first, so a new shape never spawns
// hidden inside another one.
//   center:    [x, z], the middle of the view
//   occupied:  [{ x, z, r }], objects already on the floor (r = radius from above)
//   footprint: the new shape's radius from above
// Pure math, no three.js, so it is unit tested.
export function freeSpot({ center, occupied = [], footprint = 0.5 }) {
  const clamp = (v) => Math.min(FLOOR_LIMIT, Math.max(-FLOOR_LIMIT, Number.isFinite(v) ? v : 0));
  // Snapped to a quarter unit so positions read cleanly in the Transform panel.
  const snap = (v) => Math.round(v * 4) / 4;
  const cx = snap(clamp(center?.[0]));
  const cz = snap(clamp(center?.[1]));
  const isFree = (x, z) => occupied.every((o) => Math.hypot(x - o.x, z - o.z) >= o.r + footprint + GAP);
  if (isFree(cx, cz)) return [cx, cz];

  // Rings every half unit, with spots about half a unit apart along each
  // ring, so the first free spot found is close to the nearest one.
  const STEP = 0.5;
  for (let ring = 1; ring <= 40; ring++) {
    const radius = ring * STEP;
    const count = Math.max(8, Math.round((2 * Math.PI * radius) / STEP));
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      const x = snap(clamp(cx + Math.cos(a) * radius));
      const z = snap(clamp(cz + Math.sin(a) * radius));
      if (isFree(x, z)) return [x, z];
    }
  }
  return [cx, cz]; // a packed scene: stack at the center rather than fail
}
