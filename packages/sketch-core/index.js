// Sketch solver and parametric feature history

export class SketchSolver {
  constructor() {
    this.constraints = new Map();
    this.geometryElements = new Map();
    this.featureHistory = [];
    this.solverState = 'initialized';
  }

  // Add a point (2D coordinate)
  addPoint(id, x, y) {
    this.geometryElements.set(id, {
      type: 'point',
      x,
      y,
      id
    });
    
    return { success: true, id };
  }

  // Add a line (two points)
  addLine(id, startPointId, endPointId) {
    // Verify start and end points exist
    if (!this.geometryElements.has(startPointId) || !this.geometryElements.has(endPointId)) {
      return { success: false, error: 'Endpoint points must exist for line' };
    }
    
    this.geometryElements.set(id, {
      type: 'line',
      startPoint: startPointId,
      endPoint: endPointId,
      id
    });
    
    return { success: true, id };
  }

  // Add a circle (center point and radius)
  addCircle(id, centerPointId, radius) {
    if (!this.geometryElements.has(centerPointId)) {
      return { success: false, error: 'Center point must exist for circle' };
    }
    
    this.geometryElements.set(id, {
      type: 'circle',
      center: centerPointId,
      radius,
      id
    });
    
    return { success: true, id };
  }

  // Add a constraint (e.g., horizontal, vertical, parallel, etc.)
  addConstraint(type, ...elementIds) {
    const constraint = {
      id: Date.now().toString(),
      type,
      elements: elementIds,
      createdAt: new Date()
    };
    
    this.constraints.set(constraint.id, constraint);
    
    // For demo purposes, we'll validate some constraints
    const validConstraints = ['horizontal', 'vertical', 'parallel', 'perpendicular', 'coincident', 'distance'];
    if (!validConstraints.includes(type)) {
      console.warn(`Unrecognized constraint type: ${type}`);
    }
    
    return { success: true, id: constraint.id };
  }

  // Solve the sketch (placeholder)
  async solve() {
    if (this.solverState === 'solving') {
      return { success: false, error: 'Solver already running' };
    }
    
    this.solverState = 'solving';
    
    try {
      // In a real implementation, this would run the constraint solver
      await new Promise(resolve => setTimeout(resolve, 10)); // Simulate async work
      
      this.solverState = 'solved';
      
      return {
        success: true,
        solvedAt: new Date(),
        constraints: Array.from(this.constraints.values()).length,
        elements: Array.from(this.geometryElements.values()).length
      };
    } catch (error) {
      this.solverState = 'error';
      return { success: false, error: error.message };
    }
  }

  // Add a feature to the history (like extrude, fillet, etc.)
  addFeatureToHistory(operation, parameters, result, dependencies) {
    const feature = {
      id: Date.now().toString(),
      operation,
      parameters,
      result,
      dependencies,
      timestamp: new Date(),
      projectId: this.getActiveProjectId()
    };
    
    this.featureHistory.push(feature);
    
    // Keep only the last 100 features to avoid memory issues
    if (this.featureHistory.length > 100) {
      this.featureHistory.shift();
    }
    
    return { success: true, id: feature.id };
  }

  // Get feature history
  getFeatureHistory() {
    return Array.from(this.featureHistory);
  }

  // Get current workspace state
  getCurrentState() {
    return {
      constraints: Array.from(this.constraints.values()),
      geometryElements: Array.from(this.geometryElements.values()),
      solverState: this.solverState,
      featureCount: this.featureHistory.length
    };
  }

  // Clear workspace
  clear() {
    this.constraints.clear();
    this.geometryElements.clear();
    this.solverState = 'initialized';
    return { success: true };
  }

  // Set active project (for context tracking)
  setActiveProject(projectId) {
    this.activeProject = projectId;
  }

  getActiveProjectId() {
    return this.activeProject;
  }
}

// Export singleton instance
export const sketchSolver = new SketchSolver();

// Feature types that can be created in Phase 2 (in a real implementation, these would be more complex)
export const FEATURE_TYPES = {
  SKETCH: 'sketch',
  EXTRUDE: 'extrude', 
  REVOLVE: 'revolve',
  FILLET: 'fillet',
  CHAMFER: 'chamfer',
  HOLE: 'hole',
  LINEAR_PATTERN: 'linear_pattern',
  BOOLEAN_OP: 'boolean_operation'
};

// Constraint types
export const CONSTRAINT_TYPES = {
  COINCIDENT: 'coincident',
  HORIZONTAL: 'horizontal', 
  VERTICAL: 'vertical',
  PARALLEL: 'parallel',
  PERPENDICULAR: 'perpendicular',
  DISTANCE: 'distance',
  ANGLE: 'angle'
};