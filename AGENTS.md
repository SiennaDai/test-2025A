# 2025A Modeling Instructions

## Scope

This directory contains the 2025 CUMCM Problem A project. Modeling work may read the repository's public APIs and recipes, but all event-specific assumptions, code, configuration, and outputs must stay below `competitions/2025A/`.

## Source order

1. Treat `data/raw/CUMCM-2025-problem+A-Chinese.pdf` as the authoritative problem statement.
2. Use `problem.md` as the visually checked structured transcription.
3. Use `data/processed/problem-extracted.md` only as extraction evidence, not as an interpreted model.
4. Do not use online solutions before producing an independent formulation and a Question 1 benchmark.

## Collaboration workflow

- Work one checkpoint at a time; do not solve all five questions in one pass.
- Record facts, assumptions, derived quantities, user choices, and unresolved unknowns separately.
- Update `analysis-log.md` after each agreed checkpoint.
- Maintain the current formal specification in `model-spec.md`.
- Ask the user before adopting a consequential interpretation that the problem does not state.
- Begin with geometry and kinematics, then obtain an independently reproducible result for Question 1 before designing optimization for Questions 2–5.

## Write boundaries

- Modeling Agent may modify `problem.md`, `model-spec.md`, `analysis-log.md`, `configs/`, `src/`, `run.py`, and generated `outputs/` in this directory.
- Treat `data/raw/` as read-only.
- Do not modify repository-level `src/mmkit`, `recipes`, dependencies, or tests. Report a reusable gap to the Coding Agent with a mathematical contract and minimal local example.
- Keep generated figures, tables, logs, and manifests under `outputs/`.

## Required checks

- Define coordinate system, units, time origin, trajectories, visibility criterion, and interval-union convention explicitly.
- Check dimensions, signs, domains, physical feasibility, and boundary cases.
- Separate numerical correctness from model validity.
- Preserve random seeds, solver status, tolerances, inputs, and configuration in formal runs.
