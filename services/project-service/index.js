// Project service - a simple SQLite-based storage service for Phase 1

import { projectManager } from '@berkelium/project-core';
import { sceneManager } from '@berkelium/scene-core';

export class ProjectService {
  constructor() {
    this.projects = new Map();
    this.activeProjectId = null;
  }

  // Create a project
  async createProject(name, path) {
    const project = projectManager.createProject(name, path);
    this.activeProjectId = project.id;
    
    // Initialize basic project structure
    await this.initializeProjectStructure(project.id);
    
    return project;
  }

  // Initialize basic project structure on creation
  async initializeProjectStructure(projectId) {
    // Create project directories
    const fs = require('fs');
    const path = require('path');
    
    // For demonstration purposes
    console.log(`Initializing project structure for project ${projectId}`);
    
    return { success: true };
  }

  // Get current project info
  getCurrentProject() {
    if (this.activeProjectId) {
      return projectManager.getProject(this.activeProjectId);
    }
    return null;
  }

  // Load a scene
  async loadScene(sceneId) {
    const scene = sceneManager.loadScene(sceneId);
    return scene;
  }

  // Save project manifest
  async saveManifest(projectId, manifestData) {
    return projectManager.saveManifest(projectId, manifestData);
  }

  // Get project metadata
  getProjectMetadata(projectId) {
    const project = projectManager.getProject(projectId);
    if (project && project.manifest) {
      return project.manifest;
    }
    return null;
  }

  // Check if a path is valid for this project
  validateProjectPath(projectId, relativePath) {
    try {
      // In-phase-1 implementation, just check that it's not accessing parent directories
      const fs = require('fs');
      const path = require('path');
      
      // This is a very basic validation - actual implementation would be more robust
      if (relativePath.includes('../') || relativePath.includes('..\\')) {
        return { valid: false, error: 'Invalid path: contains parent directory references' };
      }
      
      return { valid: true, error: null };
    } catch (error) {
      return { valid: false, error: error.message };
    }
  }
}

export const projectService = new ProjectService();