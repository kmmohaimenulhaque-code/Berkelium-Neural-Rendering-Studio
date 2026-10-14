// defaultScene.js
// What a new Berkelium scene contains. The ids are stable on purpose: the
// viewport uses them to find the matching three.js object for each row.

export const DEFAULT_SCENE = [
  {
    id: "scene",
    name: "Scene",
    type: "scene",
    selectable: false,
    hideable: false,
    children: [
      { id: "camera", name: "Camera", type: "camera", hideable: false },
      { id: "cube", name: "Cube", type: "mesh" },
      { id: "grid", name: "Grid", type: "helper" },
      { id: "sun", name: "Directional Light", type: "light" },
      { id: "sky", name: "Sky Light", type: "light" },
    ],
  },
];
