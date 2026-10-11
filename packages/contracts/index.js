// Application contracts for IPC and service validation

// Project creation request
export const PROJECT_CREATE_REQUEST = {
  type: 'PROJECT_CREATE',
  schema: {
    project: {
      name: { type: 'string', required: true },
      path: { type: 'string', required: true },
      version: { type: 'string', required: true }
    }
  }
};

// Scene loading request
export const SCENE_LOAD_REQUEST = {
  type: 'SCENE_LOAD',
  schema: {
    projectId: { type: 'string', required: true },
    sceneId: { type: 'string', required: true }
  }
};

// File reading request
export const FILE_READ_REQUEST = {
  type: 'FILE_READ',
  schema: {
    projectId: { type: 'string', required: true },
    filePath: { type: 'string', required: true }
  }
};

// Query request to model
export const QUERY_MODEL_REQUEST = {
  type: 'QUERY_MODEL',
  schema: {
    prompt: { type: 'string', required: true },
    context: { type: 'object', required: false }
  }
};

// Permission check request
export const PERMISSION_CHECK_REQUEST = {
  type: 'PERMISSION_CHECK',
  schema: {
    action: { type: 'string', required: true },
    target: { type: 'string', required: false }
  }
};

// Validation function for requests
export function validateRequest(request, contract) {
  if (!request || !contract.schema) {
    return { valid: false, error: 'Invalid request or contract' };
  }

  const schema = contract.schema;
  const errors = [];

  // Check required fields
  for (const [field, fieldDef] of Object.entries(schema)) {
    if (fieldDef.required && (request[field] === undefined || request[field] === null)) {
      errors.push(`Missing required field: ${field}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors: errors
  };
}