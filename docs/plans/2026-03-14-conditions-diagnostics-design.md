# Conditions & Diagnostics Design

**Date**: 2026-03-14
**Topic**: cl-acorn sections 4.2 (structured error handling) and 4.4 (introspection/diagnostics)
**Status**: Approved

## Goal

Implement structured error handling (CL conditions/restarts) and post-inference diagnostics
for the `cl-acorn.inference` package, satisfying concept.md sections 4.2 and 4.4.

## Approach

New file `src/conditions.lisp` defines all condition types. Inference functions gain
restarts and return an `inference-diagnostics` struct as an extra return value.

## File Changes

```
cl-acorn/
├── src/
│   ├── conditions.lisp          ← NEW: all condition/restart definitions
│   ├── distributions/
│   │   ├── package.lisp          (add :export for new condition types)
│   │   └── *.lisp                (raise model-errors instead of plain errors)
│   └── inference/
│       ├── package.lisp          (add :export for diagnostics struct + conditions)
│       ├── hmc.lisp              (add restarts + return diagnostics)
│       ├── nuts.lisp             (add restarts + return diagnostics w/ n-divergences)
│       └── vi.lisp               (add restarts + return diagnostics)
├── tests/
│   ├── conditions-test.lisp      ← NEW
│   └── inference-diagnostics-test.lisp  ← NEW
└── cl-acorn.asd                  (add conditions.lisp before distributions/inference)
```

**Load order**: `conditions.lisp` must be loaded before `distributions/` and `inference/`.

**Backward compatibility**: existing `(values samples accept-rate)` becomes
`(values samples accept-rate diagnostics)`. CL ignores extra values, so existing
`multiple-value-bind` with 2 vars continues to work unchanged.

## Section 4.2: Condition Hierarchy

### Base condition

```lisp
(define-condition acorn-error (error)
  ((message :initarg :message :reader acorn-error-message))
  (:report (lambda (c s) (format s "~A" (acorn-error-message c)))))
```

### Model errors (distribution / log-pdf problems)

```lisp
(define-condition model-error (acorn-error) ())

(define-condition invalid-parameter-error (model-error)
  ((parameter :initarg :parameter :reader invalid-parameter-error-parameter)
   (value     :initarg :value     :reader invalid-parameter-error-value))
  (:report (lambda (c s)
    (format s "Invalid parameter ~A = ~A: ~A"
            (invalid-parameter-error-parameter c)
            (invalid-parameter-error-value c)
            (acorn-error-message c)))))

(define-condition log-pdf-domain-error (model-error)
  ((distribution :initarg :distribution :reader log-pdf-domain-error-distribution))
  (:report (lambda (c s)
    (format s "~A: ~A" (log-pdf-domain-error-distribution c)
            (acorn-error-message c)))))
```

### Inference errors (sampling / optimization problems)

```lisp
(define-condition inference-error (acorn-error) ())

(define-condition invalid-initial-params-error (inference-error)
  ((params :initarg :params :reader invalid-initial-params-error-params))
  (:report (lambda (c s)
    (format s "Invalid initial params ~A: ~A"
            (invalid-initial-params-error-params c)
            (acorn-error-message c)))))

(define-condition non-finite-gradient-error (inference-error)
  ((params :initarg :params :reader non-finite-gradient-error-params))
  (:report (lambda (c s)
    (format s "Non-finite gradient at params ~A: ~A"
            (non-finite-gradient-error-params c)
            (acorn-error-message c)))))

(define-condition high-divergence-warning (warning)
  ((n-divergences :initarg :n-divergences :reader high-divergence-warning-n-divergences)
   (n-samples     :initarg :n-samples     :reader high-divergence-warning-n-samples))
  (:report (lambda (c s)
    (format s "High divergence rate: ~A/~A transitions diverged"
            (high-divergence-warning-n-divergences c)
            (high-divergence-warning-n-samples c)))))
```

### Signal policy

- `+log-pdf-sentinel+` is kept for AD gradient continuity
- σ≤0 and similar caller mistakes → `invalid-parameter-error`
- Current `assert` / plain `error` calls → replaced with appropriate condition signals

## Section 4.2: Restarts

### HMC / NUTS: invalid initial params

```lisp
(restart-case
  (progn
    (unless (finite-p initial-lp)
      (error 'invalid-initial-params-error
             :params initial-params
             :message "Initial log-posterior is not finite")))
  (use-fallback-params (new-params)
    :report "Supply new initial params and retry"
    (hmc log-posterior new-params ...))
  (return-empty-samples ()
    :report "Return empty sample list"
    (values '() 0.0d0 (make-empty-diagnostics))))
```

### NUTS: high divergence rate

```lisp
(when (> divergence-rate +high-divergence-threshold+)   ; threshold: 0.1 (10%)
  (restart-case
    (warn 'high-divergence-warning
          :n-divergences n-divergences :n-samples n-collected)
    (continue-with-warnings ()
      :report "Continue sampling despite high divergences"
      nil)))
```

### VI: same `use-fallback-params` / `return-empty-samples` pattern for initial mu/sigma

## Section 4.4: Diagnostics Struct

```lisp
(defstruct inference-diagnostics
  "Post-inference summary statistics returned by HMC, NUTS, and VI."
  (accept-rate       0.0d0 :type double-float)
  (n-divergences     0     :type (integer 0))
  (final-step-size   0.0d0 :type double-float)
  (n-samples         0     :type (integer 0))
  (n-warmup          0     :type (integer 0))
  (elapsed-seconds   0.0d0 :type double-float))
```

### Return value changes

| Function     | Before                                    | After                                               |
|--------------|-------------------------------------------|-----------------------------------------------------|
| `infer:hmc`  | `(values samples accept-rate)`            | `(values samples accept-rate diagnostics)`          |
| `infer:nuts` | `(values samples accept-rate)`            | `(values samples accept-rate diagnostics)`          |
| `infer:vi`   | `(values mu-list sigma-list elbo-history)`| `(values mu-list sigma-list elbo-history diagnostics)` |

### Usage example

```lisp
(multiple-value-bind (samples accept-rate diag)
    (infer:hmc #'log-posterior '(0.0d0 0.0d0) :n-samples 1000)
  (format t "Accept rate:    ~,1F%~%" (* 100 accept-rate))
  (format t "Divergences:    ~A~%"    (infer:diagnostics-n-divergences diag))
  (format t "Final step size:~,4F~%"  (infer:diagnostics-final-step-size diag))
  (format t "Elapsed:        ~,2Fs~%" (infer:diagnostics-elapsed-seconds diag)))
```

## Testing

### tests/conditions-test.lisp

- `invalid-parameter-error` is signaled for non-finite initial log-posterior
- `use-fallback-params` restart recovers and returns valid samples
- `return-empty-samples` restart returns empty list without error
- `high-divergence-warning` is signaled on pathological log-posterior (NUTS)
- `continue-with-warnings` restart allows sampling to proceed

### tests/inference-diagnostics-test.lisp

- `hmc` returns `inference-diagnostics` struct as 3rd value
- `nuts` returns `inference-diagnostics` struct as 3rd value
- `vi` returns `inference-diagnostics` struct as 4th value
- `diagnostics-n-samples` / `diagnostics-n-warmup` match `:n-samples` / `:n-warmup` args
- `diagnostics-accept-rate` is in [0, 1]
- `diagnostics-n-divergences` ≥ 0
- existing 2-value `multiple-value-bind` still compiles and works (backward compat)

## Success Criteria

- All 170 existing tests continue to pass
- New condition-test and diagnostics-test suites pass
- `infer:hmc`, `infer:nuts`, `infer:vi` signal correct condition types
- Restarts are invocable via `handler-bind` / `invoke-restart`
- Examples (09/10/11) still run without modification
