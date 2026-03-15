(defpackage #:cl-acorn.inference
  (:nicknames #:infer)
  (:use #:cl)
  (:export
   ;; Inference algorithms
   #:hmc
   #:nuts
   #:vi
   ;; Base condition types
   #:acorn-error
   #:acorn-error-message
   #:model-error
   #:inference-error
   ;; Model error subtypes
   #:invalid-parameter-error
   #:invalid-parameter-error-parameter
   #:invalid-parameter-error-value
   #:log-pdf-domain-error
   #:log-pdf-domain-error-distribution
   ;; Inference error subtypes
   #:invalid-initial-params-error
   #:invalid-initial-params-error-params
   #:non-finite-gradient-error
   #:non-finite-gradient-error-params
   #:non-finite-gradient-error-message
   ;; Warnings
   #:high-divergence-warning
   #:high-divergence-warning-n-divergences
   #:high-divergence-warning-n-samples
   ;; Restart names
   #:use-fallback-params
   #:return-empty-samples
   #:continue-with-warnings
   ;; Diagnostics struct + accessors (make-inference-diagnostics / copy-inference-diagnostics are internal)
   #:inference-diagnostics
   #:inference-diagnostics-p
   #:diagnostics-accept-rate
   #:diagnostics-n-divergences
   #:diagnostics-final-step-size
   #:diagnostics-n-samples
   #:diagnostics-n-warmup
   #:diagnostics-elapsed-seconds))
