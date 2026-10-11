// AI agent reliability and autonomy policies

export class AIAutonomy {
  constructor() {
    this.policies = new Map();
    this.agentStates = new Map();
    this.actionLog = [];
    this.reliabilityMetrics = {
      uptime: 0,
      successRate: 0,
      errorRate: 0,
      responseTime: 0
    };
  }

  // Create a new autonomy policy
  async createPolicy(name, settings) {
    const policy = {
      id: `policy_${Date.now()}`,
      name,
      createdAt: new Date(),
      updatedAt: new Date(),
      settings: {
        maxAutonomyLevel: 'read_only',
        allowedTools: [],
        approvalRequired: false,
        timeout: 30000, // 30 seconds default
        safetyChecks: true,
        ...settings
      },
      enabled: true,
      version: '1.0'
    };

    this.policies.set(policy.id, policy);
    
    return policy;
  }

  // Set autonomy level for an agent
  async setAutonomyLevel(agentId, level) {
    const validLevels = ['read_only', 'plan_generation', 'approval_required', 
                        'restricted_auto', 'explicit_privileged'];
    
    if (!validLevels.includes(level)) {
      throw new Error(`Invalid autonomy level: ${level}`);
    }

    // Simulate setting the agent's autonomy
    await new Promise(resolve => setTimeout(resolve, 50));
    
    const state = this.agentStates.get(agentId) || {
      id: agentId,
      createdAt: new Date(),
      autonomyLevel: 'read_only',
      lastUpdated: new Date()
    };
    
    state.autonomyLevel = level;
    state.lastUpdated = new Date();
    
    this.agentStates.set(agentId, state);
    
    return { success: true, level };
  }

  // Get current agent autonomy level
  getAgentAutonomyLevel(agentId) {
    const state = this.agentStates.get(agentId);
    return state ? state.autonomyLevel : 'read_only';
  }

  // Validate tool usage against policy
  async validateToolUsage(agentId, toolName, parameters) {
    const policy = this.getActivePolicy(agentId);
    
    if (!policy || !policy.enabled) {
      return { allowed: false, reason: 'No active policy' };
    }
    
    // Check if tool is allowed for current autonomy level
    const toolAllowed = this.isToolAllowed(policy, toolName);
    
    if (!toolAllowed) {
      return { 
        allowed: false, 
        reason: `Tool ${toolName} not permitted at current autonomy level` 
      };
    }
    
    // Run safety checks if enabled
    if (policy.settings.safetyChecks) {
      const safetyCheck = await this.runSafetyChecks(agentId, toolName, parameters);
      if (!safetyCheck.passed) {
        return { 
          allowed: false, 
          reason: `Safety check failed: ${safetyCheck.reason}` 
        };
      }
    }
    
    // Validate parameters
    const paramCheck = await this.validateParameters(toolName, parameters);
    if (!paramCheck.valid) {
      return { 
        allowed: false, 
        reason: `Invalid parameters: ${paramCheck.reason}` 
      };
    }
    
    return { allowed: true, tool: toolName };
  }

  // Run safety checks on proposed action
  async runSafetyChecks(agentId, toolName, parameters) {
    // In a real implementation this would verify:
    // - File path restrictions
    // - Resource usage limits
    // - Project boundary constraints
    // - Data integrity checks
    await new Promise(resolve => setTimeout(resolve, 50));
    
    return {
      passed: Math.random() > 0.1, // 90% chance of passing for demo
      checks: [
        {
          name: 'file_access',
          passed: true,
          details: 'Access to project files within scope'
        },
        {
          name: 'resource_usage',
          passed: true,
          details: 'Operation within resource limits'
        },
        {
          name: 'data_integrity',
          passed: true,
          details: 'No risk of data corruption'
        }
      ]
    };
  }

  // Validate tool parameters
  async validateParameters(toolName, parameters) {
    // In a real implementation this would check:
    // - Required vs optional parameters
    // - Valid ranges for numeric parameters
    // - Format validation for strings
    await new Promise(resolve => setTimeout(resolve, 30));
    
    return { 
      valid: true,
      parameters, 
      validatedAt: new Date() 
    };
  }

  // Apply an autonomy policy to an agent
  async applyPolicy(agentId, policyId) {
    const policy = this.policies.get(policyId);
    if (!policy) {
      throw new Error(`Policy ${policyId} not found`);
    }
    
    // For demo purposes, we'll just log the application
    await new Promise(resolve => setTimeout(resolve, 50));
    
    return {
      success: true,
      agentId,
      policyId,
      appliedAt: new Date()
    };
  }

  // Get active policy for agent
  getActivePolicy(agentId) {
    // In a real implementation this would reference policy assignments
    // For demo, we'll simulate with the first policy if one exists
    const policy = Array.from(this.policies.values())[0];
    return policy || null;
  }

  // Is tool allowed for current policy?
  isToolAllowed(policy, toolName) {
    // Check the autonomy level and allowed tools
    switch (policy.settings.maxAutonomyLevel) {
      case 'read_only':
        return ['inspect_scene', 'search_code', 'explain_file'].includes(toolName);
      
      case 'plan_generation':
        return ['generate_plan', 'inspect_scene', 'search_code'].includes(toolName);
      
      case 'approval_required':
        return ['request_cad_operation', 'run_tests', 'create_scene_object'].includes(toolName);
      
      case 'restricted_auto':
        return ['apply_patch', 'start_job', 'export_representation'].includes(toolName);
      
      case 'explicit_privileged':
        // All tools allowed, but still require approval for mutating operations
        return true;
      
      default:
        return false;
    }
  }

  // Log an AI action
  logAction(agentId, action, details) {
    const entry = {
      id: `log_${Date.now()}`,
      agentId,
      timestamp: new Date(),
      action,
      details,
      status: 'executed'
    };
    
    this.actionLog.push(entry);
    
    // Keep only recent logs
    if (this.actionLog.length > 1000) {
      this.actionLog.shift();
    }
    
    return entry;
  }

  // Generate reliability metrics
  getReliabilityMetrics() {
    // In a real implementation, this would analyze historical data
    const timestamp = new Date();
    
    return {
      id: `metrics_${Date.now()}`,
      timestamp,
      uptime: `${(Math.random() * 100).toFixed(2)}%`,
      successRate: `${(Math.random() * 30 + 70).toFixed(1)}%`,
      errorRate: `${(Math.random() * 5).toFixed(1)}%`,
      averageResponseTime: `${(Math.random() * 500 + 100).toFixed(0)}ms`,
      lastUpdated: timestamp
    };
  }

  // Update reliability metrics based on recent activity
  updateMetrics(agentId, success = true) {
    // Simulate updating metrics from action results
    return this.getReliabilityMetrics();
  }

  // Get agent history
  getAgentHistory(agentId) {
    return this.actionLog.filter(log => log.agentId === agentId)
      .slice(-10); // Last 10 actions
  }

  // Get all policies
  getAllPolicies() {
    return Array.from(this.policies.values());
  }

  // Enable/disable a policy
  async togglePolicy(policyId, enabled) {
    const policy = this.policies.get(policyId);
    if (!policy) return { success: false, error: 'Policy not found' };
    
    policy.enabled = enabled;
    policy.updatedAt = new Date();
    
    return { 
      success: true, 
      policyId, 
      enabled,
      updatedAt: policy.updatedAt
    };
  }

  // Create default autonomy profile
  async createDefaultProfile(agentId) {
    const policy = await this.createPolicy('Default Agent Profile', {
      maxAutonomyLevel: 'approval_required',
      allowedTools: ['request_cad_operation', 'run_tests', 'create_scene_object'],
      approvalRequired: true,
      timeout: 30000,
      safetyChecks: true
    });
    
    await this.applyPolicy(agentId, policy.id);
    
    return policy;
  }

  // Get supported autonomy levels
  getSupportedLevels() {
    return [
      {
        level: 'read_only',
        description: 'Agent can only inspect and read data',
        tools: ['inspect_scene', 'search_code', 'explain_file']
      },
      {
        level: 'plan_generation',
        description: 'Agent can generate plans and propose operations', 
        tools: ['generate_plan', 'inspect_scene', 'search_code']
      },
      {
        level: 'approval_required',
        description: 'Agent can request operations that require approval',
        tools: ['request_cad_operation', 'run_tests', 'create_scene_object']
      },
      {
        level: 'restricted_auto',
        description: 'Agent can execute operations without approval (limited)',
        tools: ['apply_patch', 'start_job', 'export_representation']
      },
      {
        level: 'explicit_privileged',
        description: 'Agent with full autonomy but explicit permissions',
        tools: ['all'] // All tools require explicit approval
      }
    ];
  }

  // Get current agent state summary
  getAgentState(agentId) {
    const state = this.agentStates.get(agentId) || { 
      id: agentId, 
      autonomyLevel: 'read_only' 
    };
    
    return {
      ...state,
      policies: Array.from(this.policies.values()).filter(p => p.enabled),
      reliabilityMetrics: this.getReliabilityMetrics()
    };
  }

  // Execute an approved action
  async executeAction(agentId, toolName, parameters) {
    // Validate the execution is allowed
    const validation = await this.validateToolUsage(agentId, toolName, parameters);
    
    if (!validation.allowed) {
      return { 
        success: false, 
        error: `Action not permitted: ${validation.reason}` 
      };
    }
    
    // Simulate actual action execution
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const success = Math.random() > 0.1; // 90% success rate for demo
    
    if (success) {
      this.updateMetrics(agentId, true);
      this.logAction(agentId, toolName, { parameters, status: 'completed' });
      
      return {
        success: true,
        result: `Successfully performed ${toolName} with specified parameters`,
        timestamp: new Date()
      };
    } else {
      this.updateMetrics(agentId, false);
      this.logAction(agentId, toolName, { parameters, status: 'failed' });
      
      return {
        success: false,
        error: `Failed to perform ${toolName} - execution error`,
        timestamp: new Date()
      };
    }
  }
}

// Export singleton instance
export const aiAutonomy = new AIAutonomy();

// Autonomy levels
export const AUTONOMY_LEVELS = {
  READ_ONLY: 'read_only',
  PLAN_GENERATION: 'plan_generation', 
  APPROVAL_REQUIRED: 'approval_required',
  RESTRICTED_AUTO: 'restricted_auto',
  EXPLICIT_PRIVILEGED: 'explicit_privileged'
};

// Policy settings
export const POLICY_SETTINGS = {
  MAX_AUTONOMY_LEVEL: 'maxAutonomyLevel',
  ALLOWED_TOOLS: 'allowedTools',
  APPROVAL_REQUIRED: 'approvalRequired',
  TIMEOUT: 'timeout',
  SAFETY_CHECKS: 'safetyChecks'
};