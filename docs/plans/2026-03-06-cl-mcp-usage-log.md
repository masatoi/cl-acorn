# cl-mcp Usage Log — Examples Implementation

Recording tool usage patterns, friction points, and issues encountered during example implementation.

## Legend
- **SMOOTH**: Tool worked as expected, good experience
- **FRICTION**: Tool worked but required workaround or was awkward
- **ISSUE**: Tool failed or produced unexpected result
- **TIP**: Learned something useful about tool usage

---

## Observations

### Task 1: Newton-Raphson (03-newton-method)

**Tools used**: `fs-get-project-info`, `lisp-read-file` (collapsed+full), `repl-eval`, `fs-write-file`, `lisp-check-parens`

- **FRICTION**: `(load "examples/03-newton-method/main.lisp")` via `repl-eval` failed because REPL's CWD didn't match project root. Had to use absolute path. This is a recurring pattern — always use absolute paths in `repl-eval`.
- **FRICTION**: `repl-eval` captures return values but printed output (format to `*standard-output*`) goes to REPL stdout, not tool response. Had to wrap in `with-output-to-string` to verify iteration tables. This is expected but adds friction for output-heavy examples.
- **SMOOTH**: `lisp-read-file` collapsed view gave quick API orientation without full file reads.
- **SMOOTH**: `lisp-check-parens` gave fast paren validation before load attempt.
- **TIP**: Prototype AD calls in `repl-eval` before writing file catches issues early (confirmed `ad:expt`, `ad:cos` etc. work).

### Task 2: Sensitivity Analysis (04-sensitivity)

**Tools used**: `lisp-read-file`, `lisp-check-parens`, `repl-eval`, `fs-write-file`

- **SMOOTH**: No friction at all. All tools worked as expected on first try.
- **SMOOTH**: `lisp-read-file` collapsed view efficient for surveying transcendental.lisp API.
- **SMOOTH**: `repl-eval` + `with-output-to-string` pattern now established — AD matches analytical to 0.0-4.4e-16 error.
- **TIP**: New files → `fs-write-file` is appropriate (not `lisp-edit-form`). Absolute paths in `repl-eval` now habitual.

### Task 3: Black-Scholes Greeks (05-black-scholes)

**Tools used**: `load-system`, `repl-eval` (extensive prototyping), `lisp-read-file`, `fs-write-file`, `fs-get-project-info`, `sequential-thinking`

- **ISSUE**: **Nested `derivative` doesn't work** — `derivative` calls `(make-dual x 1.0d0)` which coerces x to `double-float`. When x is already a dual from an outer `derivative`, coercion silently strips the epsilon. True nested derivatives (hyper-duals) are not supported. Gamma was computed via central difference on AD-computed Delta instead.
- **SMOOTH**: REPL-first prototyping caught the nested derivative issue early (before writing full file).
- **SMOOTH**: `sequential-thinking` used to analyze the nested derivative limitation and plan workaround.
- **TIP**: The `norm-cdf` sign branch needs `(typep x 'ad:dual)` check to extract real part for comparison. Pattern: `(if (typep x 'ad:dual) (minusp (ad:dual-real x)) (minusp x))`.

### Task 4: Curve Fitting (01-curve-fitting) [parallel]

**Tools used**: `load-system`, `repl-eval`, `lisp-read-file`, `fs-write-file`

- **ISSUE**: **`LOOP ... SUM` doesn't work with dual numbers** — CL's `SUM` loop clause uses `CL:+` internally, which signals a type error on dual numbers. Must use manual accumulation with `ad:+` and `setf` instead.
- **FRICTION**: **PLN not available in `repl-eval`** — Package-local-nicknames defined via `(:local-nicknames ...)` in defpackage aren't available when prototyping in `repl-eval` because the reader resolves packages at read time. Workaround: use full `cl-acorn.ad:` prefix during REPL prototyping.
- **ISSUE**: **FORMAT `~` in strings** — `~` in format strings is interpreted as directives. Need `~~` to escape.
- **SMOOTH**: REPL-first prototyping caught the LOOP SUM bug early.

### Task 6: PID Control (06-pid-control) [parallel]

**Tools used**: `fs-get-project-info`, `lisp-read-file`, `repl-eval`

- **SMOOTH**: No issues. AD propagated through 500-step Euler simulation loop without problems.
- **SMOOTH**: `lisp-read-file` collapsed → full view workflow efficient for API exploration.
- **TIP**: The `ad:` arithmetic handles mixed dual/number operations transparently through the entire simulation loop.

### Task 7: Signal Processing (07-signal-processing) [parallel]

**Tools used**: `fs-get-project-info`, `lisp-read-file`, `repl-eval`, `lisp-check-parens`

- **FRICTION**: **Package symbol interning between test loads** — When reloading an example package in `repl-eval`, symbols from a previous read can get interned in the wrong package. Workaround: delete and recreate the package between test runs.
- **SMOOTH**: `lisp-check-parens` gave pre-commit confidence.
- **SMOOTH**: Convention consistency across examples made following patterns easy.

### Task 5: Neural Network (02-neural-network)

**Tools used**: `fs-write-file`, `lisp-check-parens`, `lisp-read-file`, `repl-eval`, `fs-get-project-info`

- **ISSUE**: **`CL:PI` naming conflict** — Using `pi` as a lambda parameter shadows `CL:PI` constant, causing subtle bugs. Renamed to `param-i`.
- **SMOOTH**: REPL-driven development caught the PI conflict and allowed iterative hyperparameter tuning without leaving the loop.
- **SMOOTH**: `lisp-check-parens` useful for verifying large data files with many nested parens.
- **TIP**: For performance-sensitive examples, use a data subset (30 samples) with forward-mode AD. 67 params × 100 epochs = 6700 derivative calls is tractable but slow.
- **TIP**: `with-output-to-string` + `repl-eval` is the reliable pattern for validating example output.
