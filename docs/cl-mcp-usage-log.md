# cl-mcp Usage Experience Log

This file accumulates feedback on using cl-mcp tools during development.

## Session: 2026-03-13 — Inference Building Blocks Implementation

### Summary
Implemented 3 new sub-packages (distributions, optimizers, inference) with 10 tasks using subagent-driven development. All 136 tests pass.

### Tool Usage Patterns

**`load-system` + `run-tests`**: Primary workflow for verify-after-edit. Used dozens of times across subagents. Reliable with `force=true` default. `clear_fasls=true` useful for full regression.

**`lisp-edit-form` / `lisp-patch-form`**: Used effectively for `cl-acorn.asd` modifications and implementation files that use standard CL symbols. **Cannot be used** for files referencing package-local-nicknames (e.g., `ad:`, `dist:`, `opt:`, `infer:` prefixes) because the CST parser doesn't resolve them. This is the primary friction point.

**Workaround**: For files with package nicknames, subagents used the standard `Write` / `Edit` tools instead. This works but loses the structural safety guarantees of `lisp-edit-form`.

### Issues & Friction

1. **Package nickname resolution in CST parser** (Known limitation)
   - `lisp-edit-form` fails on files using `ad:+`, `dist:normal-log-pdf` etc.
   - All test files and most implementation files use nicknames
   - Workaround: Use `Write`/`Edit` tools for these files
   - Impact: High — most files in this project are affected

2. **`defconstant` with array values** (SBCL-specific)
   - `(defconstant +foo+ #(...))` fails on reload because arrays aren't `eql`
   - Solution: Use `defvar` instead for non-scalar constants
   - Not a cl-mcp issue, but relevant for CL development guidance

3. **`pi` as lambda parameter** (CL-specific)
   - `(lambda (pi ...) ...)` fails because `cl:pi` is a constant
   - Not a cl-mcp issue, but the plan template used `pi` as a variable name
   - Solution: Rename to `pv` or `p-i`

### Positive Notes

- `run-tests` with structured output (pass/fail counts) is excellent for CI-style workflows
- `load-system` with `force=true` reliably picks up file changes
- `repl-eval` useful for quick verification during implementation
- Pool architecture (parent for file ops, worker for eval) works transparently
- `clgrep-search` effective for finding symbols without loading systems

### Suggestion

Consider adding an option to `lisp-edit-form` that accepts a list of package nicknames to resolve, e.g., `"nicknames": {"ad": "cl-acorn.ad", "dist": "cl-acorn.distributions"}`. This would eliminate the primary friction point without requiring changes to the CST parser itself.
