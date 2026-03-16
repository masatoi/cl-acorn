# AGENTS.md

This file provides guidance to coding agents working in this repository.

This repository is intended to be developed with `cl-mcp` tooling. Prefer the
Common Lisp MCP tools for exploration, editing, evaluation, and testing over
ad hoc shell-driven workflows.

## Required cl-mcp Workflow For Lisp Files

  Before Lisp file operations, call `fs-set-project-root` with the current project root.

  For `.lisp` and `.asd` files:
  - use `lisp-read-file` for reading
  - use `clgrep-search` for project-wide search
  - use `load-system` for loading or reloading ASDF systems
  - use `repl-eval` for small experiments and immediate verification
  - use `run-tests` for test execution

  For existing `.lisp` files:
  - you MUST use `lisp-edit-form` or `lisp-patch-form` for edits
  - do NOT use `fs-write-file` on existing Lisp source files
  - do NOT use text-based patching tools for existing Lisp source unless the structured tools are unavailable

  Follow this loop for substantial Lisp changes:
  1. Explore the relevant code and tests
  2. Experiment with small forms in the REPL
  3. Persist the minimal correct change with structured editing tools
  4. Verify with targeted tests, then broader tests if needed

  After editing a Lisp file, explicitly reload the affected system with `load-system` or re-evaluate the changed forms. File edits are not automatically
  visible to the worker process.

  When creating a new Lisp file:
  - write a minimal valid file first
  - verify structure with `lisp-check-parens` if needed
  - then extend it using `lisp-edit-form`

## Agent Guidelines

Read these files before making substantial Common Lisp changes:

- `prompts/repl-driven-development.md` (required: load and follow this workflow)
- `agents/common-lisp-expert.md`

### Required `cl-mcp` Workflow For Lisp Work

Before Lisp file operations, call `fs-set-project-root` with the current project
root.

Use this loop for substantial Common Lisp changes:

1. Explore the relevant package, tests, and examples.
2. Experiment with small forms in the REPL before making larger edits.
3. Persist the minimal correct change.
4. Verify with targeted tests, then broader tests if the surface area grew.

For `.lisp` and `.asd` files:

- use `lisp-read-file` for reading
- use `clgrep-search` for project-wide search
- use `load-system` for loading or reloading ASDF systems
- use `repl-eval` for small experiments and immediate verification
- use `run-tests` for test execution

For existing `.lisp` files:

- you MUST use `lisp-edit-form` or `lisp-patch-form` for edits
- do NOT use `fs-write-file` on existing Lisp source files
- do NOT use text-based patching tools for existing Lisp source unless the
  structured tools are unavailable

After editing a Lisp file, explicitly reload the affected system with
`load-system` or re-evaluate the changed forms. File edits are not automatically
visible to the worker process.

When creating a new Lisp file:

- write a minimal valid file first
- verify structure with `lisp-check-parens` if needed
- then extend it using `lisp-edit-form`

## Project Overview

`cl-acorn` is a Common Lisp library for:

- forward-mode automatic differentiation via dual numbers
- reverse-mode automatic differentiation via a dynamic tape
- probability distributions with AD-friendly log-density functions
- optimizers such as SGD and Adam
- Bayesian inference building blocks including HMC, NUTS, and VI

The project aims to stay lightweight and portable. The main system has no
runtime dependencies beyond ANSI Common Lisp. Tests use Rove.

## Build And Development Commands

### Load The System

```lisp
(ql:quickload :cl-acorn)
;; or
(asdf:load-system :cl-acorn)
```

### Run Tests

Agents should use `cl-mcp` `run-tests` by default:

```json
{"system":"cl-acorn/tests"}
```

Use targeted runs whenever possible:

```json
{"system":"cl-acorn/tests","test":"cl-acorn/tests/gradient-test::test-gradient-quadratic"}
```

Shell `rove` runs are a fallback for humans or when `run-tests` is unavailable:

```bash
rove cl-acorn.asd
```

Or from a REPL:

```lisp
(asdf:test-system :cl-acorn)
```

### Load An Example

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

Most examples are standalone and intended to be loaded directly this way.

## Source Layout

### Core AD

- `src/package.lisp`: exports for `cl-acorn.ad` (`ad`)
- `src/dual.lisp`: dual number representation
- `src/arithmetic.lisp`, `src/transcendental.lisp`: forward-mode operators
- `src/interface.lisp`: public forward-mode entry point such as `derivative`
- `src/tape.lisp`: reverse-mode tape nodes and backprop support
- `src/reverse-arithmetic.lisp`, `src/reverse-transcendental.lisp`: reverse-mode operators
- `src/gradient.lisp`: `gradient`, `jacobian-vector-product`, and `hessian-vector-product`

### Probability And Optimization

- `src/distributions/`: `cl-acorn.distributions` (`dist`) package and log-pdf/sample implementations
- `src/optimizers/`: `cl-acorn.optimizers` (`opt`) package with SGD and Adam

### Inference

- `src/inference/package.lisp`: exports for `cl-acorn.inference` (`infer`)
- `src/inference/conditions.lisp`: condition hierarchy, warnings, diagnostics
- `src/inference/hmc.lisp`, `src/inference/nuts.lisp`, `src/inference/vi.lisp`: inference algorithms
- `src/inference/dual-avg.lisp`: dual averaging utilities for adaptation

### Tests, Examples, Benchmarks

- `tests/`: Rove suites covering AD, distributions, optimizers, validation, and inference
- `examples/`: runnable demonstrations of optimization, simulation, and inference workflows
- `benchmarks/cl/`: benchmark system entries

## Important Patterns And Constraints

### Preserve Package Boundaries

Public APIs are split across these packages:

- `cl-acorn.ad` (`ad`)
- `cl-acorn.distributions` (`dist`)
- `cl-acorn.optimizers` (`opt`)
- `cl-acorn.inference` (`infer`)

When adding or changing public functionality:

- update the relevant `defpackage` exports
- keep package nicknames stable
- update README examples if the user-facing API changed

### Keep ASDF In Sync

`cl-acorn.asd` uses `:serial t` in several modules. If you add a new file:

- place it in the correct load order
- add it to the right module in `cl-acorn.asd`
- add matching test files to the test system when needed

Do not assume ASDF will discover new files automatically.

### Prefer `cl-mcp` Tools Over Shell Commands

For Common Lisp development in this repository:

- prefer `lisp-read-file` over raw file dumps for Lisp source
- prefer `clgrep-search` for project-wide symbol and pattern search
- prefer `load-system` over manual `asdf:load-system` during agent work
- prefer `run-tests` over shelling out to `rove`

Use shell commands mainly for git, simple repository inspection, or when the
user explicitly requests them.

### Record `cl-mcp` Feedback

When implementation work uses `cl-mcp`, accumulate concise feedback in
`docs/cl-mcp-feedback/`.

- treat `docs/cl-mcp-feedback/` as the canonical location for new `cl-mcp`
  usage notes
- create or update a dated Markdown file for the task or session, for example
  `YYYY-MM-DD-short-topic.md`
- capture concrete friction points, successful usage patterns, workarounds, and
  suggestions that would improve the tool or workflow
- keep entries short and specific so later agents can reuse the feedback during
  similar work

### Respect Numeric Conventions

This codebase consistently coerces numeric inputs and outputs to
`double-float` at API boundaries. Preserve that behavior unless there is a
strong reason to change it, and then update tests accordingly.

Be careful not to introduce plain CL arithmetic where AD dispatch is required.
Functions intended to be differentiable should use `ad:+`, `ad:*`, `ad:sin`,
and related operators internally.

### Conditions Over Ad Hoc Failures

Inference and validation code already uses explicit condition types such as:

- `infer:invalid-parameter-error`
- `infer:invalid-initial-params-error`
- `infer:non-finite-gradient-error`
- `infer:high-divergence-warning`

Prefer extending this condition-based approach over introducing generic
unstructured errors for new failure modes.

### Tests Should Cover Behavior And Error Paths

When changing numerical or inference code, add or update tests for:

- nominal behavior
- edge cases and invalid inputs
- condition signaling and warnings
- diagnostics or return-shape changes

For probabilistic algorithms, prefer assertions on robust properties such as
finite outputs, acceptance-rate ranges, approximate sample means, or the
presence of diagnostics rather than brittle exact values.

### Keep Examples And Docs Honest

The examples directory and `README.md` are part of the user-facing surface.
If a change affects public APIs or expected workflows, update the relevant
example or documentation in the same change.

## Suggested Agent Workflow

1. Read the relevant source file, matching tests, and any affected example.
2. Check the relevant package exports before editing public APIs.
3. Make the smallest change that satisfies the request.
4. Run targeted tests first.
5. Prefer `run-tests` for both targeted and full-suite verification.
6. Fall back to `rove cl-acorn.asd` only if `run-tests` is unavailable or insufficient.

## Notes For Common Tasks

### Adding A New Public Function

- implement it in the appropriate package
- export it from the corresponding `package.lisp`
- add tests in `tests/`
- update `README.md` if users are expected to call it directly

### Adding A New Distribution Or Optimizer

- keep the public API consistent with existing naming conventions
- add both behavior tests and validation tests
- wire the file into `cl-acorn.asd`

### Changing Inference Code

- review `src/inference/conditions.lisp` first
- preserve diagnostics structure unless the change explicitly redesigns it
- verify both happy-path sampling and failure/validation behavior
