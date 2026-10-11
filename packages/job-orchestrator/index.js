// Job orchestration system for CPU/GPU-intensive work

export class JobOrchestrator {
  constructor() {
    this.jobs = new Map();
    this.jobQueue = [];
    this.isRunning = false;
    this.activeWorkers = new Set();
    this.backendCapabilities = new Map();
  }

  // Initialize the job system
  async initialize() {
    console.log('Initializing job orchestration system...');
    
    // Simulate backend capability detection
    this.backendCapabilities.set('cpu', { 
      available: true, 
      cores: 8,
      memory: '16GB',
      architecture: 'x64'
    });
    
    this.backendCapabilities.set('gpu', {
      available: false,  // We'll set to true when appropriate
      type: 'none',
      memory: '0MB',
      capabilities: []
    });
    
    this.isRunning = true;
    return { success: true, message: 'Job orchestration system initialized' };
  }

  // Submit a new job
  async submitJob(jobType, inputRefs, options = {}) {
    if (!this.isRunning) {
      throw new Error('Job orchestrator not initialized');
    }

    const job = {
      id: Date.now().toString(),
      type: jobType,
      status: 'pending',
      submittedAt: new Date(),
      inputRefs,
      options,
      priority: options.priority || 'normal',
      progress: 0,
      logs: [],
      attempts: 0
    };

    // Validate job type
    const supportedTypes = ['scene_processing', 'photogrammetry', 'neural_rendering_training', 
                           'rendering_conversion', 'scientific_processing', 'build_and_test'];
    
    if (!supportedTypes.includes(jobType)) {
      throw new Error(`Unsupported job type: ${jobType}`);
    }

    this.jobs.set(job.id, job);
    this.jobQueue.push(job.id);

    // For testing purposes, we'll process it immediately
    await this.processJob(job.id);
    
    return {
      success: true,
      id: job.id,
      status: job.status
    };
  }

  // Process a job (simulating async processing)
  async processJob(jobId) {
    if (!this.jobs.has(jobId)) {
      throw new Error(`Job ${jobId} not found`);
    }

    const job = this.jobs.get(jobId);
    
    // Update status to running
    job.status = 'running';
    job.startedAt = new Date();
    this.logJobEvent(jobId, `Job started: ${job.type}`);

    try {
      // Simulate job processing with different time intervals based on job type
      let durationMs;
      switch (job.type) {
        case 'science_processing':
          durationMs = 2000; // Fast for science processing
          break;
        case 'rendering_conversion':
          durationMs = 3000; // Medium for rendering conversion
          break;
        case 'neural_rendering_training':
          durationMs = 5000; // Long for training
          break;
        default:
          durationMs = 1000; // Default
      }

      // Simulate progress updates
      for (let i = 0; i < 10; i++) {
        await new Promise(resolve => setTimeout(resolve, durationMs / 10));
        job.progress = Math.min(100, (i + 1) * 10);
        this.logJobEvent(jobId, `Progress: ${job.progress}%`);
      }

      // Finalize job
      job.status = 'completed';
      job.completedAt = new Date();
      this.logJobEvent(jobId, 'Job completed successfully');
      
      return { success: true, id: jobId };
    } catch (error) {
      job.status = 'failed';
      job.failedAt = new Date();
      job.error = error.message;
      this.logJobEvent(jobId, `Job failed: ${error.message}`);
      throw error;
    }
  }

  // Get a specific job
  getJob(jobId) {
    return this.jobs.get(jobId);
  }

  // Get all jobs
  getAllJobs() {
    return Array.from(this.jobs.values());
  }

  // Get jobs by type
  getJobsByType(type) {
    return Array.from(this.jobs.values()).filter(job => job.type === type);
  }

  // Cancel a job
  async cancelJob(jobId) {
    if (!this.jobs.has(jobId)) {
      return { success: false, error: `Job ${jobId} not found` };
    }

    const job = this.jobs.get(jobId);
    
    // Only allow cancellation of running or pending jobs
    if (job.status === 'completed' || job.status === 'failed') {
      return { success: false, error: `Cannot cancel completed/failed job` };
    }
    
    job.status = 'cancelled';
    job.cancelledAt = new Date();
    this.logJobEvent(jobId, 'Job cancelled by user');

    return { success: true, id: jobId };
  }

  // Get job status
  getJobStatus(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    
    return {
      id: job.id,
      type: job.type,
      status: job.status,
      progress: job.progress,
      submittedAt: job.submittedAt,
      startedAt: job.startedAt,
      completedAt: job.completedAt,
      error: job.error
    };
  }

  // Add a log entry for a job
  logJobEvent(jobId, message) {
    if (!this.jobs.has(jobId)) {
      return;
    }
    
    const job = this.jobs.get(jobId);
    const timestamp = new Date();
    job.logs.push({
      timestamp,
      message
    });
  }

  // Get job logs
  getJobLogs(jobId) {
    if (!this.jobs.has(jobId)) {
      return [];
    }
    
    const job = this.jobs.get(jobId);
    return job.logs;
  }

  // Get system capabilities
  getSystemCapabilities() {
    return {
      isRunning: this.isRunning,
      jobsCount: this.jobs.size,
      pendingJobsCount: Array.from(this.jobs.values()).filter(j => j.status === 'pending').length,
      runningJobsCount: Array.from(this.jobs.values()).filter(j => j.status === 'running').length,
      backendCapabilities: Object.fromEntries(this.backendCapabilities)
    };
  }

  // Set GPU capabilities 
  setGPUInfo(gpuInfo) {
    this.backendCapabilities.set('gpu', {
      available: true,
      type: gpuInfo.type,
      memory: gpuInfo.memory,
      capabilities: gpuInfo.capabilities || []
    });
    
    return { success: true, message: 'GPU information updated' };
  }

  // Check if job can be checkpointed
  canCheckpoint(jobType) {
    // In a real implementation, this would check specific job type capabilities
    const checkpointableTypes = ['neural_rendering_training', 'photogrammetry'];
    
    return checkpointableTypes.includes(jobType);
  }

  // Simulate resuming job from checkpoint (placeholder)
  async resumeFromCheckpoint(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }
    
    if (job.status !== 'checkpointed') {
      throw new Error('Job is not in checkpointed state');
    }

    // Simulate resumption process
    await new Promise(resolve => setTimeout(resolve, 100));
    
    job.status = 'resumed';
    return { success: true, id: jobId };
  }

  // Get all pending jobs (for queue processing)
  getPendingJobs() {
    return Array.from(this.jobs.values())
      .filter(job => job.status === 'pending')
      .sort((a, b) => {
        // Prioritize by priority and then by submission time
        const priorityOrder = { high: 3, normal: 2, low: 1 };
        if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
          return priorityOrder[b.priority] - priorityOrder[a.priority];
        }
        return a.submittedAt - b.submittedAt;
      });
  }

  // Process job queue
  async processQueue() {
    const jobs = this.getPendingJobs();
    
    for (const job of jobs) {
      try {
        await this.processJob(job.id);
      } catch (error) {
        console.error(`Failed to process job ${job.id}:`, error.message);
      }
    }
  }
}

// Export singleton instance
export const jobOrchestrator = new JobOrchestrator();

// Supported job types
export const JOB_TYPES = {
  SCENE_PROCESSING: 'scene_processing',
  PHOTOGRAMMETRY: 'photogrammetry',
  NEURAL_RENDERING_TRAINING: 'neural_rendering_training',
  RENDERING_CONVERSION: 'rendering_conversion',
  SCIENTIFIC_PROCESSING: 'scientific_processing',
  BUILD_AND_TEST: 'build_and_test'
};

// Job statuses
export const JOB_STATUSES = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
  CHECKPOINTED: 'checkpointed',
  RESUMED: 'resumed'
};

// Job priorities
export const JOB_PRIORITIES = {
  HIGH: 'high',
  NORMAL: 'normal', 
  LOW: 'low'
};