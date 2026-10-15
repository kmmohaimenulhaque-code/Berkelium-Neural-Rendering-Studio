// defaultScene.js
// What a new Berkelium scene contains. The ids are stable on purpose: the
// viewport uses them to find the matching three.js object for each row.
//
// Only the cube can be moved or deleted. The camera, grid and lights are part
// of the studio itself, so deleting them by accident would break the view.

export const STARTER_CUBE_ID = "cube";

export const DEFAULT_SCENE = [
  {
    id: "scene",
    name: "Scene",
    type: "scene",
    selectable: false,
    hideable: false,
    removable: false,
    children: [
      { id: "camera", name: "Camera", type: "camera", hideable: false, removable: false },
      {
        id: STARTER_CUBE_ID,
        name: "Cube",
        type: "mesh",
        transformable: true,
        transform: { p: [0, 0.6, 0], r: [0, 0, 0], s: [1, 1, 1] },
      },
      { id: "grid", name: "Grid", type: "helper", removable: false },
      { id: "sun", name: "Directional Light", type: "light", removable: false },
      { id: "sky", name: "Sky Light", type: "light", removable: false },
    ],
  },
];
