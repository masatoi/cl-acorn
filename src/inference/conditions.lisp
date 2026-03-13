(in-package #:cl-acorn.inference)

;;;; Condition hierarchy and diagnostics struct for cl-acorn.

;;; ---- Base error type -------------------------------------------------

(define-condition acorn-error (error)
  ((message :initarg :message :reader acorn-error-message))
  (:report (lambda (c s)
    (format s "~A" (acorn-error-message c))))
  (:documentation "Base class for all cl-acorn error conditions."))

;;; ---- Model errors (distribution / log-pdf problems) -----------------

(define-condition model-error (acorn-error) ()
  (:documentation "Supertype for errors in model or distribution evaluation."))

(define-condition invalid-parameter-error (model-error)
  ((parameter :initarg :parameter :reader invalid-parameter-error-parameter)
   (value     :initarg :value     :reader invalid-parameter-error-value))
  (:report (lambda (c s)
    (format s "Invalid parameter ~A = ~A: ~A"
            (invalid-parameter-error-parameter c)
            (invalid-parameter-error-value c)
            (acorn-error-message c))))
  (:documentation "Signaled when a distribution parameter has an invalid value."))

(define-condition log-pdf-domain-error (model-error)
  ((distribution :initarg :distribution
                 :reader log-pdf-domain-error-distribution))
  (:report (lambda (c s)
    (format s "~A: ~A"
            (log-pdf-domain-error-distribution c)
            (acorn-error-message c))))
  (:documentation "Signaled when log-pdf is evaluated outside its support."))

;;; ---- Inference errors (sampling / optimization problems) ------------

(define-condition inference-error (acorn-error) ()
  (:documentation "Supertype for errors in inference algorithms."))

(define-condition invalid-initial-params-error (inference-error)
  ((params :initarg :params :reader invalid-initial-params-error-params))
  (:report (lambda (c s)
    (format s "Invalid initial params ~A: ~A"
            (invalid-initial-params-error-params c)
            (acorn-error-message c))))
  (:documentation
   "Signaled when initial parameters produce a non-finite log-probability."))

(define-condition non-finite-gradient-error (inference-error)
  ((params :initarg :params :reader non-finite-gradient-error-params))
  (:report (lambda (c s)
    (format s "Non-finite gradient at params ~A: ~A"
            (non-finite-gradient-error-params c)
            (acorn-error-message c))))
  (:documentation "Signaled when the gradient is non-finite at a parameter set."))

;;; ---- Warnings --------------------------------------------------------

(define-condition high-divergence-warning (warning)
  ((n-divergences :initarg :n-divergences
                  :reader high-divergence-warning-n-divergences)
   (n-samples     :initarg :n-samples
                  :reader high-divergence-warning-n-samples))
  (:report (lambda (c s)
    (format s "High divergence rate: ~A/~A transitions diverged (~,1F%). ~
               Consider reducing step-size or reparameterizing the model."
            (high-divergence-warning-n-divergences c)
            (high-divergence-warning-n-samples c)
            (* 100.0d0
               (/ (float (high-divergence-warning-n-divergences c))
                  (float (max 1 (high-divergence-warning-n-samples c))))))))
  (:documentation
   "Warned when NUTS post-warmup divergence rate exceeds the threshold."))

;;; ---- Post-inference diagnostics struct -------------------------------

(defstruct (inference-diagnostics (:conc-name diagnostics-))
  "Post-inference summary statistics returned by HMC, NUTS, and VI."
  (accept-rate     0.0d0 :type double-float)
  (n-divergences   0     :type (integer 0))
  (final-step-size 0.0d0 :type double-float)
  (n-samples       0     :type (integer 0))
  (n-warmup        0     :type (integer 0))
  (elapsed-seconds 0.0d0 :type double-float))
