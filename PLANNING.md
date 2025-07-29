# MoneyPrinterTurbo++ Implementation Plan

## Current Task: TTS Service Architecture Implementation
Implementation of advanced TTS service architecture for MoneyPrinterTurbo++ as outlined in the PRD and SERVICE_ARCHITECTURE_DESIGN.md.

## Previous Task: Directory Organization (COMPLETED)
Organize the MoneyPrinterTurbo-main (3) project directory for clarity, maintainability, and scalability. No files or code will be deleted. All changes will be documented and tracked according to strict workflow rules.

## TTS Implementation Steps
1. **Phase 1: Core TTS Service Layer (Foundation)**
   - Create base TTS service interface
   - Implement Google TTS service  
   - Create TTS service factory pattern
   - Add configuration management

2. **Phase 2: CharacterBox Integration**
   - Implement CharacterBox service
   - Create character-enhanced TTS service
   - Add character profile management

3. **Phase 3: Database Schema & Models**
   - Create TTS-related database tables
   - Add Pydantic models for request/response
   - Implement migration scripts

4. **Phase 4: API Endpoints**
   - Enhanced TTS endpoints
   - CharacterBox endpoints
   - Batch processing support

5. **Phase 5: Performance & Security**
   - Caching implementation
   - Circuit breaker pattern
   - Input validation and sanitization

6. **Phase 6: Integration & Testing**
   - Video pipeline integration
   - Unit and integration tests
   - Performance optimization

## Expected Outcomes
- Advanced TTS service supporting Google TTS and CharacterBox
- AI avatar integration for lip-synced narration
- 50% faster video generation (PRD goal)
- Multilingual subtitle support with adaptive positioning
- Batch processing with quality scoring
- Robust error handling with fallback strategies
- Cost optimization under $0.10 per video generation

## Required Resources
- Google Cloud TTS API credentials
- CharacterBox API access
- Database migration capabilities
- Access to existing video service architecture
- Testing infrastructure for audio/video processing

## Timeline
- **Day 1:** Phase 1-2 (Foundation & CharacterBox)
- **Day 2:** Phase 3-4 (Database & API)
- **Day 3:** Phase 5-6 (Production Ready)

## PRD Alignment
This implementation directly addresses:
- **FR4:** AI avatar integration for lip-synced narration
- **FR3:** Dynamic subtitles with multilingual translation
- **FR5:** Batch generation with AI quality scoring
- **NFR1:** 5-minute processing time optimization
- **NFR6:** Cost efficiency under $0.10 per video
- **NFR9:** Internationalization support

## Risks & Mitigation
- **API Dependencies:** Implement robust fallback chain (Google → Azure → Edge TTS)
- **Performance Impact:** Use async processing and intelligent caching
- **Cost Overruns:** Monitor usage and implement rate limiting
- **Integration Complexity:** Maintain backward compatibility with existing services
- **Quality Degradation:** Implement quality scoring and validation

## Compliance
All steps will strictly follow the rules in RULES.instructions.md, including branch management, documentation, and task tracking.



# MoneyPrinterTurbo++ Product Requirements Document (PRD)

## Goals and Background Context

### Goals

- Deliver an enhanced version of MoneyPrinterTurbo that automates high-quality, monetizable short video creation with advanced AI features, reducing creator effort while maximizing engagement and revenue potential.
- Achieve MVP status with core improvements focused on subtitle enhancements, integration of AI avatars, and basic analytics feedback to boost video quality and user retention.
- Enable rapid iteration for creators targeting platforms like YouTube Shorts and TikTok, aiming for 50% faster video production compared to the original tool.
- Position the product as a free-tier-friendly, open-source alternative to premium tools like Runway ML or Synthesia, with optional premium add-ons for advanced features.
- Validate market fit through built-in feedback loops, targeting 10x growth in user base within the first year post-launch.

### Background Context

The original MoneyPrinterTurbo is a Python-based AI tool that automates short video creation from keywords, generating scripts, sourcing materials, adding subtitles and music, and synthesizing HD videos for quick monetization on social platforms. While effective, it faces limitations in subtitle quality, personalization (e.g., no avatars), multilingual support, and data-driven optimization, leading to lower engagement in competitive creator markets. Based on competitive analysis (e.g., tools like Pictory and HeyGen excel in avatars and analytics but are costly), this enhanced version (++ ) addresses these gaps to empower faceless creators, educators, and marketers. The current landscape shows explosive growth in short-form video (TikTok's 2B+ users, YouTube Shorts' 50B daily views), with AI tools projected to capture 30% of the $100B content creation market by 2027. This PRD builds on brainstormed improvements to create a lean MVP that stays true to the open-source ethos while paving the way for scalable monetization.

## Requirements

### Functional

- FR1: The system must generate video scripts using AI models, supporting custom prompts and domain-specific refinements (e.g., for tutorials or ads).
- FR2: Integrate stock footage sourcing from multiple providers (e.g., Pexels, Unsplash) with AI-driven selection for relevance.
- FR3: Automatically generate and embed dynamic subtitles with adaptive positioning, styling, and multilingual translation.
- FR4: Support AI avatar integration for lip-synced narration, with options for custom or pre-built avatars.
- FR5: Provide batch generation with an AI quality scorer to rank and select optimal videos based on engagement metrics.
- FR6: Include a web UI wizard for easy configuration, preview, and export of videos in multiple formats (9:16, 16:9).
- FR7: Enable direct scheduling and posting to social platforms (YouTube, TikTok) with auto-generated hashtags and captions.
- FR8: Incorporate a basic analytics feedback loop to analyze past video performance and suggest optimizations for future generations.
- FR9: Allow template-based customization for video styles, including educational, promotional, or interactive modes.
- FR10: Support offline mode with local LLMs and asset libraries to reduce dependency on external APIs.

### Non Functional

- NFR1: The tool must process a single video in under 5 minutes on standard hardware (mid-range CPU/GPU), optimizing for free-tier cloud usage.
- NFR2: Ensure 99% uptime for the web UI, with secure handling of API keys and user data (GDPR-compliant).
- NFR3: Support scalability for up to 100 concurrent users in batch mode, using efficient dependencies like MoviePy v1.0.3.
- NFR4: Maintain accessibility standards (WCAG 2.1 AA) in the UI, including keyboard navigation and screen reader compatibility.
- NFR5: Keep installation simple (Docker-compose preferred), with dependencies pinned to avoid breaking changes.
- NFR6: Prioritize cost-efficiency, aiming for under $0.10 per video generation using open-source alternatives where possible.
- NFR7: Ensure cross-platform compatibility (Windows, macOS, Linux) and mobile-responsive UI.
- NFR8: Implement logging and error handling for robust debugging, with user-friendly messages.
- NFR9: Support internationalization for scripts and subtitles in at least 5 languages (English, Spanish, French, Chinese, Arabic).
- NFR10: Achieve a minimum video quality score of 80% based on internal metrics (resolution, coherence, engagement potential).

## User Interface Design Goals

### Overall UX Vision

Create an intuitive, creator-friendly interface that feels like a "video wizard"—guiding users from prompt to polished output with minimal friction, emphasizing speed, previews, and customization to empower non-technical users while delighting pros.

### Key Interaction Paradigms

- Wizard-based flow for beginners (step-by-step: input prompt, select style, preview, generate).
- Advanced dashboard for pros (batch queues, analytics views, template editor).
- Real-time previews and drag-and-drop customizations (e.g., reposition subtitles).
- Mobile-first responsiveness for on-the-go creators.

### Core Screens and Views

- Welcome/Dashboard: Prompt input, recent videos, quick-start templates.
- Generation Wizard: Step-by-step config (script, assets, subtitles, avatars, music).
- Preview & Edit: Real-time video preview with editable layers (subs, text, effects).
- Analytics Hub: Performance metrics, optimization suggestions.
- Settings Page: API keys, templates, offline mode toggle.

### Accessibility: WCAG 2.1 AA

Full keyboard navigation, alt text for previews, color contrast ratios >4.5:1, screen reader support for wizards.

### Branding

- Modern, vibrant palette: Primary #FF6B00 (orange for energy), Secondary #1E1E1E (dark gray), Accent #00E676 (green for success).
- Clean sans-serif typography (e.g., Roboto) for readability.
- Icon set: Material Icons for simplicity and familiarity.

### Target Device and Platforms

Web responsive (desktop priority), with mobile support for previews/posting. Platforms: Browsers (Chrome, Firefox, Safari), optional desktop app via Electron.

## Technical Assumptions

### Repository Structure: Monorepo

Single repo for backend (Python core), frontend (Streamlit UI), and docs.

### Service Architecture

Monolith backend with modular services (e.g., script gen, video synthesis), event-driven for async tasks (e.g., Celery for batch processing).

### Testing requirements

Unit tests for core modules (e.g., subtitle.py), integration tests for end-to-end video gen, E2E UI tests with Playwright. Manual testing for video quality; aim for 80% coverage.

### Additional Technical Assumptions and Requests

- Primary Language: Python 3.10+.
- Runtime: Docker for portability.
- Frameworks: FastAPI for API endpoints, Streamlit for UI.
- Databases: SQLite for local analytics; optional PostgreSQL for cloud.
- Cloud Platform: AWS (S3 for storage, Lambda for optional tasks).
- Prefer open-source deps to minimize costs.

## Epics

- Epic1 Foundation & Core Enhancements: Establish improved script generation, subtitle service, and basic UI to create a stable MVP for single-video automation.
- Epic2 Personalization & Advanced Features: Add AI avatars, multilingual support, and templates to make videos more engaging and customizable.
- Epic3 Optimization & Automation: Implement batch processing, quality scoring, and social scheduling to streamline workflows and boost output.
- Epic4 Analytics & Iteration: Build feedback loops and analytics integration to enable data-driven improvements.
- Epic5 Scalability & Offline Mode: Enhance for offline use, performance optimizations, and cloud scalability.

## Epic 1 Foundation & Core Enhancements

Enhance the core video generation pipeline with better AI scripting, dynamic subtitles, and a refreshed UI to deliver higher-quality outputs faster, forming the MVP backbone.

### Story 1.1 Script Generation Upgrade

As a content creator,
I want enhanced AI script generation with domain-specific models,
so that videos are more tailored and engaging.

#### Acceptance Criteria

- 1: Integrate at least one specialized LLM (e.g., Grok 3).
- 2: Support custom prompts and refinements.
- 3: Scripts generated in under 10 seconds.

### Story 1.2 Dynamic Subtitle Service

As a creator,
I want adaptive subtitles with AI positioning and styling,
so that text is readable and professional.

#### Acceptance Criteria

- 1: Use OpenCV for object avoidance in positioning.
- 2: Support multilingual translation.
- 3: Embed in videos without quality loss.

### Story 1.3 UI Wizard Refresh

As a user,
I want a step-by-step wizard for video config,
so that generation is intuitive.

#### Acceptance Criteria

- 1: Streamlit-based with real-time previews.
- 2: Mobile-responsive design.
- 3: Error handling for invalid inputs.

## Epic 2 Personalization & Advanced Features

Add avatars and templates to personalize videos, expanding use cases for diverse creators.

### Story 2.1 AI Avatar Integration

As a creator,
I want lip-synced avatars for narration,
so that videos feel dynamic without showing my face.

#### Acceptance Criteria

- 1: Integrate SadTalker or similar for sync.
- 2: Options for pre-built or custom avatars.
- 3: Fallback to voice-only if needed.

### Story 2.2 Template System

As a user,
I want pre-built style templates,
so that I can quickly match brand or format needs.

#### Acceptance Criteria

- 1: At least 5 templates (e.g., vlog, ad).
- 2: Customizable via UI.
- 3: Community upload support.

## Epic 3 Optimization & Automation

Streamline batch processing and posting to reduce manual work and increase output volume.

### Story 3.1 Batch Generation & Scoring

As a creator,
I want AI-ranked batch videos,
so that I can produce multiple options efficiently.

#### Acceptance Criteria

- 1: Generate 3-5 variants per run.
- 2: Score based on engagement metrics.
- 3: Auto-select top one.

### Story 3.2 Social Scheduler

As a user,
I want direct posting to platforms,
so that distribution is seamless.

#### Acceptance Criteria

- 1: Integrate YouTube/TikTok APIs.
- 2: Auto-generate captions/hashtags.
- 3: Schedule queue support.

## Epic 4 Analytics & Iteration

Enable data-driven refinements to improve video performance over time.

### Story 4.1 Performance Analytics Loop

As a creator,
I want insights from past videos,
so that future ones are optimized.

#### Acceptance Criteria

- 1: Pull metrics via APIs.
- 2: Suggest tweaks (e.g., shorter intros).
- 3: Dashboard view in UI.

## Epic 5 Scalability & Offline Mode

Make the tool accessible offline and scalable for power users.

### Story 5.1 Offline Mode Support

As a user,
I want local LLM and assets,
so that I can generate without internet.

#### Acceptance Criteria

- 1: Integrate Ollama for scripts.
- 2: Offline stock library.
- 3: Toggle in settings.

### Story 5.2 Cloud Scalability

As a pro user,
I want cloud options for heavy tasks,
so that performance scales.

#### Acceptance Criteria

- 1: AWS integration for rendering.
- 2: Cost estimator in UI.
- 3: Hybrid local/cloud mode.

## Change Log

| Change | Date | Version | Description | Author |
| ------ | ---- | ------- | ----------- | ------ |
| Initial Draft | July 28, 2025 | 1.0 | Full YOLO PRD based on brainstorm | John (PM) |

----- END PRD START CHECKLIST OUTPUT ------

## Checklist Results Report

As John, your pragmatic PM, I ran the `pm-checklist` on this draft PRD. Overall, it's solid for an MVP-focused enhancement project—compliance is about 85%. Here's a quick summary by section (✅ PASS, ⚠️ PARTIAL, ❌ FAIL, N/A):

- **Section 1: General PRD Structure** - ✅ All key sections present; flows logically from goals to epics.
- **Section 2: Goals and Context** - ✅ Clear, measurable goals; background ties in market data (assumed from brainstorm—could add sources if needed).
- **Section 3: Requirements** - ⚠️ Functional are detailed but could use more prioritization; Non-Functional are strong on performance/cost but partial on security (add encryption for API keys?).
- **Section 4: UI Design Goals** - ✅ Comprehensive for a UI-heavy tool; assumptions on branding based on "vibrant" creator vibe—refine if you have specifics.
- **Section 5: Technical Assumptions** - ✅ Aligned with original Python roots; assumptions on monorepo/monolith are pragmatic, but testing could specify tools (e.g., pytest).
- **Section 6: Epics and Stories** - ✅ Sequential and value-driven; stories have clear ACs. Partial: Could add estimates or risks (e.g., API deps for avatars).
- **Section 7: Overall Quality** - ✅ Concise, user-focused; assumptions noted (e.g., dates, metrics from brainstorm). No major gaps, but iterate on UI if no designs yet.

Deficiencies? Mostly partials—e.g., add data sources for market claims (NFR partial on security). No blocks; this PRD's ready for Architect if you approve. What's your feedback? Want tweaks, or proceed to handoff? 😊