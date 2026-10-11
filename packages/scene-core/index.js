// Scene management core - placeholder for now
export class SceneManager {
  constructor() {
    this.scenes = new Map();
  }

  // Create a new scene
  createScene(projectId, name) {
    const scene = {
      id: Date.now().toString(),
      projectId,
      name,
      createdAt: new Date(),
      lastModified: new Date()
    };
    
    this.scenes.set(scene.id, scene);
    return scene;
  }

  // Load a scene
  loadScene(sceneId) {
    return this.scenes.get(sceneId);
  }

  // Save a scene
  saveScene(sceneId, data) {
    const scene = this.loadScene(sceneId);
    if (scene) {
      scene.data = data;
      scene.lastModified = new Date();
      return true;
    }
    return false;
  }
}

export const sceneManager = new SceneManager();

// Simple USD/JSON scene format for Phase 1
export function createBasicScene() {
  return {
    version: "1.0",
    scene: {
      name: "Berkelium Scene",
      metadata: {
        created: new Date().toISOString(),
        author: "Berkelium Studio"
      },
      objects: []
    }
  };
}