---
mode: "agent"
---
You are an expert in configuration management and codebase refactoring, with deep knowledge of TOML formatting, Python, Docker, TypeScript, and related tools. Your task is to aggregate all configuration and environment settings from the listed files in the MoneyPrinter codebase into a single, unified config.toml file located in the root directory. This aggregation should consolidate redundant or overlapping settings, eliminate inconsistencies, and organize everything hierarchically using TOML sections and subsections for clarity and maintainability.

Prior to processing any file, translate its contents to English if they contain any non-English text, ensuring all keys, values, comments, and descriptions are in clear, standard English while preserving technical accuracy.

Process the files sequentially, one by one, in the order provided. For each file:

1. **Analyze the File**: Review its contents, purpose, and structure. Identify key-value pairs, lists, dependencies, environment variables, build instructions, or other configurable elements that can be represented in TOML format. Note any non-configurable parts (e.g., executable scripts or imperative code) that cannot be directly aggregated; suggest handling them separately (e.g., via comments or external references). If any environment variables or settings are hardcoded in the file, replace them with variables that reference the unified config.toml file (e.g., use placeholders like '${config.toml.path.to.key}' or suggest code modifications to load dynamically from config.toml where the actual keys and values are stored).

2. **Extract and Map Settings**: 
   - Map settings to appropriate TOML sections (e.g., [environment] for .env vars, [dependencies.python] for requirements.txt, [docker.services.config] for service-specific Dockerfiles).
   - Resolve conflicts or redundancies by prioritizing the most specific or up-to-date values (e.g., merge similar env vars, deduplicate dependencies).
   - Convert formats as needed: dotenv to key-value tables, YAML/JSON to nested TOML structures, lists (e.g., requirements) to arrays.
   - Handle sensitive data by using placeholders (e.g., "SECRET_KEY = '{{ placeholder }}'").
   - For files like Dockerfiles or scripts, extract configurable parameters (e.g., ENV directives, ARG values) into TOML, while noting that the full file may need to remain separate but reference the TOML.

3. **Build the Aggregated TOML Incrementally**: Maintain a running version of the unified config.toml. After processing each file, update this aggregated file with the new mappings and provide the current state.

4. **Output for Each File**:
   - State the file path and name.
   - Summarize extracted settings and any issues (e.g., conflicts, unmergeable elements, hardcoded values replaced) in a bullet-point list.
   - Provide the updated full contents of the aggregated config.toml after incorporating this file.
   - Explain key mappings and decisions in a concise paragraph.
   - If a file contributes nothing (e.g., pure script with no configs), note it and proceed.

After processing all files, finalize the config.toml with:
- A top-level [global] section for overarching settings.
- Comments for documentation, explaining sections and origins.
- Validation for TOML syntax correctness.
- Recommendations for updating the codebase to read from this single file (e.g., load TOML in Python code, reference in Docker Compose).

Process files one at a time and wait for confirmation before the next. Use best practices: ensure the TOML is human-readable, uses consistent naming (e.g., snake_case for keys), and supports environment overrides.

Complete List of Files to Process (in this order):

**Core Configuration Files**  
- config.toml  
- config.mcp.example.toml  
- app/config/ (treat as a directory; aggregate any files within)  

**Environment Files**  
- .env.example  
- app/.env  
- deployment/environments/.env.development  

**Dependency Management**  
- requirements.txt  
- app/requirements.txt  
- app/requirements-py313.txt  
- app/mcp/clients/typescript/package.json  
- venv/lib/python3.9/site-packages/g4f/Provider/npm/package.json  

**Docker Configuration**  
- app/Dockerfile  
- app/docker-compose.yml  
- deployment/docker/docker-compose.microservices.yml  
- deployment/docker/Dockerfile.config-service  
- deployment/docker/Dockerfile.user-service  
- deployment/docker/Dockerfile.content-service  
- deployment/docker/Dockerfile.tts-service  
- deployment/docker/Dockerfile.video-service  
- deployment/docker/Dockerfile.material-service  
- deployment/docker/Dockerfile.orchestration-service  
- deployment/docker/Dockerfile.notification-service  
- deployment/docker/Dockerfile.db-migration  
- deployment/docker/Dockerfile.data-seeder  
- .dockerignore  

**Testing & CI Configuration**  
- tests/config/pytest.ini  
- tests/conftest.py  
- .github/workflows/ci.yml  
- .github/workflows/video-optimization.yml  

**Deployment Configuration**  
- northflank-deploy.yaml  
- deploy-northflank.sh  

**Development Tools**  
- app/mcp/clients/typescript/tsconfig.json  
- .gitignore  

**Special Configuration Files**  
- .config_key  
- setup_dev_environment_updated.sh  
- get-docker.sh  
- scripts/run.sh  
- scripts/setup.sh  

Begin with the first file: config.toml. Assume access to the codebase for file contents; if needed, request specific contents in your response.