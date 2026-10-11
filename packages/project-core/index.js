// Project management core
export class ProjectManager {
  constructor() {
    this.projects = new Map();
  }

  // Create a new project
  createProject(name, path) {
    const project = {
      id: Date.now().toString(),
      name,
      path,
      createdAt: new Date(),
      lastModified: new Date(),
      version: '1.0.0'
    };
    
    this.projects.set(project.id, project);
    return project;
  }

  // Get a project by ID
  getProject(id) {
    return this.projects.get(id);
  }

  // List projects
  listProjects() {
    return Array.from(this.projects.values());
  }

  // Save project manifest
  saveManifest(projectId, manifestData) {
    const project = this.getProject(projectId);
    if (project) {
      project.manifest = manifestData;
      project.lastModified = new Date();
      return true;
    }
    return false;
  }
}

export const projectManager = new ProjectManager();