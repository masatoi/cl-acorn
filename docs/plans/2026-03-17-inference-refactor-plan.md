# Inference Refactor Plan

**Goal:** Refactor `cl-acorn.inference` to reduce duplicated sampler setup logic, isolate SBCL-specific numeric helpers, and split the NUTS tree builder into smaller units without changing the public API.

**Scope:** `src/inference/package.lisp`, `src/inference/hmc.lisp`, `src/inference/nuts.lisp`, `src/inference/vi.lisp`, `cl-acorn.asd`, and new shared utility file(s) under `src/inference/`.

**Non-goals:**
- No public API redesign for `infer:hmc`, `infer:nuts`, or `infer:vi`
- No algorithmic retuning of acceptance targets or sampler defaults
- No distribution-layer refactors in this change

---

## Motivation

The current inference code duplicates the same categories of work across samplers:

- argument validation and coercion of initial parameter lists
- initial state validation and restart handling
- warning suppression and final diagnostics assembly
- SBCL-specific non-finite checks and float-trap masking

`nuts.lisp` also contains a large `build-tree` function that mixes leaf expansion,
recursive doubling, multinomial proposal selection, and statistics aggregation in
one recursive body.

This makes the inference package harder to extend and increases the risk of
behavior drift between `hmc`, `nuts`, and `vi`.

---

## Design

### 1. Shared inference support layer

Add a new internal file, tentatively `src/inference/util.lisp`, loaded after
`conditions.lisp` and before algorithm files.

This file will own:

- numeric helpers
  - `finite-double-p`
  - `every-finite-p`
  - `with-float-traps-masked`
  - any small helper needed for safe random-log / non-finite checks
- sampler argument helpers
  - validation helpers for positive integers / non-negative integers / positive reals
  - `coerce-params-to-double-float`
- top-level inference helpers
  - `resolve-initial-params`
  - `warn-high-divergence`
  - `make-final-diagnostics`

These helpers are internal to `cl-acorn.inference`; they will not be exported.

### 2. Common sampler initialization

Refactor `hmc`, `nuts`, and `vi` so that each top-level entry point becomes:

1. validate inputs through shared helpers
2. initialize shared state (`*log-pdf-error-warned-p*`, coerced params, timers)
3. run the algorithm-specific core loop
4. finalize diagnostics through a shared helper

`hmc` and `nuts` will share the same restart-driven initial parameter validation
flow via `resolve-initial-params`.

`vi` will reuse the same numeric helper layer and validation helpers, even though
its optimization loop remains separate from MCMC.

### 3. Split NUTS tree construction

Refactor `build-tree` into smaller internal helpers:

- `build-tree-leaf`
  - executes one leapfrog step
  - computes divergence / log-weight / accept statistics
  - returns a `tree-state`
- `merge-tree-states`
  - merges inner and outer subtree states
  - updates endpoints, proposal selection, and aggregate statistics
- `build-tree`
  - retains the recursive orchestration only

This keeps the current `tree-state` representation and recursive algorithm, but
reduces each function to one responsibility.

---

## Implementation Steps

### Task 1: Add shared utility file

- Update `cl-acorn.asd` to load the new inference utility file before `hmc`
- Move shared numeric helpers out of `hmc.lisp`
- Introduce common validation / coercion / diagnostics helpers

### Task 2: Refactor top-level inference entry points

- Refactor `hmc` to use the shared validation and finalization helpers
- Refactor `nuts` to use the shared validation and finalization helpers
- Refactor `vi` to use the shared validation and numeric helpers

### Task 3: Split NUTS tree logic

- Add `build-tree-leaf` and `merge-tree-states`
- Shrink `build-tree` to orchestration and recursion only
- Keep behavior and return structure unchanged

### Task 4: Verify behavior

Run targeted tests first:

- `cl-acorn/tests::hmc`
- `cl-acorn/tests::nuts`
- `cl-acorn/tests::vi`
- `cl-acorn/tests::conditions`
- `cl-acorn/tests::inference-diagnostics`

Then run the full `cl-acorn/tests` system if targeted coverage passes.

---

## Risks

- The restart-based initial parameter flow is subtle; moving it must preserve
  `use-fallback-params` and `return-empty-samples` behavior.
- NUTS proposal selection is easy to perturb accidentally; tree merge logic must
  keep the current multinomial sampling semantics.
- `with-float-traps-masked` and non-finite detection are SBCL-specific today; this
  refactor should isolate that dependency, not broaden it.

---

## Success Criteria

- Less duplicated setup logic across `hmc`, `nuts`, and `vi`
- SBCL-specific numeric helpers live in one internal file
- `nuts.lisp` no longer embeds the full tree-building algorithm in one large function
- Existing inference and diagnostics tests pass unchanged
