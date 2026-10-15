// shapeIcons.js
// One icon per shape, shared by the Add menu and the scene hierarchy so a
// sphere looks the same in both places. Kept apart from shapes.js so that
// file stays free of React.

import { Box, Circle, Cylinder, Cone, Square, Torus } from "lucide-react";

export const SHAPE_ICONS = {
  cube: Box,
  sphere: Circle,
  cylinder: Cylinder,
  cone: Cone,
  plane: Square,
  torus: Torus,
};
