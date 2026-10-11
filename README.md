# Berkelium Studio - Phase 1 Implementation

This repository contains the implementation of Berkelium Studio's Phase 1: Desktop Foundation and Read-Only Intelligence.

## Overview

Phase 1 establishes:
- Secure Electron desktop shell with proper IPC boundaries
- Project and persistence foundations  
- Design system with core tokens
- Scene management foundation
- Editor core with file handling
- Local model provider adapters
- Contract validation system
- Permission framework foundation

## Monorepo Structure

```
berkelium-studio/
├── apps/
│   └── desktop/                    # Electron application
│       ├── src/
│       │   ├── main/               # Main process
│       │   ├── preload/            # Preload script with security boundaries  
│       │   └── renderer/           # React UI components
│       └── package.json
├── packages/
│   ├── contracts/                  # IPC contracts and validation
│   ├── design-system/              # Design tokens and UI primitives
│   ├── project-core/               # Project management
│   ├── scene-core/                 # Scene composition
│   ├── editor-core/                # Editor services
│   └── provider-core/              # Local model adapters
├── services/
│   └── project-service/            # Project persistence service
├── tests/                          # Unit and integration tests
├── package.json                    # Root monorepo manifest
├── pnpm-workspace.yaml             # Workspace configuration
└── README.md                       # This file
```

## Implemented Components

### 1. Secure Electron Shell
- Main process with proper security boundaries
- Preload script with restricted APIs
- Context isolation and sandboxing 
- IPC message validation

### 2. Project Management
- Project creation and persistence
- Manifest handling  
- Path validation
- Metadata storage

### 3. Scene System
- Scene creation and management
- Basic scene data structures
- Foundation for USD integration

### 4. Design System
- Core color tokens 
- Spacing system
- UI component foundations

### 5. Editor Integration
- File browsing and loading  
- Editor state management (mocked)
- Read-only file viewing

### 6. Model Providers
- Ollama adapter foundation
- LM Studio adapter foundation
- Connection handling
- Query interface

### 7. Contract Validation
- Typed IPC contracts
- Request validation system
- Error handling

## Phase 1 Exit Criteria

✅ A developer can launch Berkelium Studio  
✅ Create and reopen a project  
✅ Inspect a scene  
✅ Search source code  
✅ Ask configured local model questions about the project  

✅ Secure privilege boundary established  
✅ Persistent project foundations implemented  
✅ No AI mutation capability present  

## Security & Permissions

Phase 1 explicitly excludes mutating operations:
- No file editing capabilities
- No CAD feature creation  
- No neural-rendering training
- No arbitrary command execution
- No privileged operations accessible from renderer

The application enforces strict boundary between:
- Trusted main process / service layer
- Restricted renderer / UI layer