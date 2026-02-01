# GEMINI.md - Agentic Orchestration Protocol

## Philosophy
This project uses a "Conductor-Performer" model. Antigravity (the orchestrator) coordinates a swarm of specialized subagents to execute high-density signal processing optimizations.

## Core Directives
1. **Tool Priority**: The `task` tool is the primary engine for implementation. Avoid manual edits for architectural changes.
2. **Parallelism**: Maximize execution throughput by firing independent `task` agents in parallel.
3. **Validation**: Every subagent output must be verified by the orchestrator before marking a dependency as resolved.

## Orchestration Patterns
- **Wave-based Execution**: Group independent tasks into "waves" that run concurrently.
- **Session Continuity**: Use `session_id` to continue complex multi-turn implementations with the same subagent.
- **Context Injection**: Provide exhaustive context in the `prompt` of `task` calls to minimize subagent uncertainty.
