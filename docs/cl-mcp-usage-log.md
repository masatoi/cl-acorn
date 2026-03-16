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

## Session: 2026-03-17 — Diagnostics Test Gap Audit

### Summary
Reviewed existing diagnostics coverage, added 4 missing tests around validation and sequence handling, and verified the full `cl-acorn/tests` suite. Final result: 242 passing tests.

### Tool Usage Patterns

**`fs-set-project-root`**: Mandatory first step. The initial failure mode was actually helpful here because the error message from `fs-get-project-info` told me exactly how to recover.

**`clgrep-search`**: Good for building a quick inventory of `src/` and `tests/` without loading systems. Broad searches over the whole tree can get verbose fast, so narrowing by directory and concern was important.

**`lisp-read-file`**: Strong default for reading `.lisp` files. Collapsed/full reads made it easy to compare exported APIs, source branches, and test coverage. On a large test file, using `name_pattern` with raw output was still less surgical than expected, so I had to rely on smaller follow-up reads.

**`lisp-edit-form`**: Worked well for inserting new `deftest` forms into an existing test file. `insert_after` was the right primitive for this task and preserved formatting cleanly.

**`repl-eval`**: Useful for probing candidate gaps before editing. I used it to confirm that `waic`/`loo` already accept vector data and to inspect the concrete condition types raised by diagnostics helpers.

**`run-tests`**: Good verification path once the worker had the system loaded. Targeted test execution was fast enough to use as a tight feedback loop before running the whole suite.

### Issues & Friction

1. **Worker discoverability after `load-system` with `force=true`**
   - After editing only a test file, `load-system` on `"cl-acorn/tests"` with `force=true` dropped into a state where the worker reported `Component "cl-acorn/tests" not found`.
   - `run-tests` then failed for the same reason until I reset the worker and re-registered the ASDF definition with `(asdf:load-asd #P"/home/wiz/cl-acorn/cl-acorn.asd")` via `repl-eval`.
   - After that, `load-system` with `force=false` and `run-tests` both worked again.
   - Impact: Medium. Recoverable, but surprising in a normal edit-test loop.

2. **Large-result ergonomics**
   - `clgrep-search` and `lisp-read-file` are both useful, but once the query is broad the result can become long enough that the next step is another narrowing pass rather than immediate action.
   - This is manageable, but it rewards incremental querying more than “one big search”.

### Positive Notes

- The parent/worker split stayed out of the way for normal read/edit/test operations.
- `lisp-check-parens` was a quick sanity check after structural edits.
- Standard package nicknames in this project (`diag:`, `infer:`, `dist:`) did not interfere with `lisp-edit-form` for this test-file edit.

### Suggestion

If `load-system` with `force=true` clears worker state that also affects ASDF discoverability for local test systems, it would help if the tool either preserved local `.asd` registration or returned a more specific remediation hint such as “run `asdf:load-asd` on the project `.asd` and retry.”

## Session: 2026-03-17 — Diagnostics Helper Coverage Audit

### Summary
Added 4 tests covering an uncovered `run-chains` validation path plus three unreferenced diagnostics helpers (`all-chain-samples`, `chain-param`, `quantile`). Final result: 246 passing tests.

### Tool Usage Patterns

**`clgrep-search`**: Effective for proving absence, not just finding code. Searching for helper names across `tests/` quickly confirmed that the candidate gaps were real instead of just hard to spot in a long file.

**`lisp-edit-form`**: Sequential `insert_after` edits were a good fit for adding small `deftest` forms without disturbing the surrounding file. For this style of work, the structural edit path felt safer than a text patch.

**`lisp-check-parens`**: Cheap verification step after multiple insertions. Good complement to `lisp-edit-form` when several edits land in the same file.

**`run-tests`**: The combination of targeted execution for just the new tests and a final full-suite run worked well. It is fast enough that there is little reason to fall back to shell-driven Rove runs here.

### Issues & Friction

1. **`repl-eval` input fragility on long multi-line forms**
   - One exploratory `repl-eval` call failed with an `END-OF-FILE` reader error because the submitted form was unbalanced.
   - The backtrace was useful, but fairly noisy for what was ultimately a simple input mistake.
   - Impact: Low. Easy to recover, but it nudges the workflow toward smaller REPL probes.

2. **Name-targeted reads still need follow-up narrowing**
   - When locating exact insertion points in a large test file, I still needed a second pass with other tools after the initial reads.
   - This is workable, but it means the “find exact anchor, then edit” loop is a little more manual than the search/edit experience.

### Positive Notes

- `fs-set-project-root` + later file operations remained predictable once initialized.
- `clgrep-search` and `run-tests` together make test-gap auditing efficient.
- `lisp-edit-form` handled this file cleanly even with project nicknames like `diag:` and `infer:`.

### Suggestion

For `repl-eval`, a lightweight preflight hint when the submitted code looks unbalanced, or a shorter default error summary before the full backtrace, would make exploratory failures easier to scan.
