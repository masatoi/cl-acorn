(in-package #:cl-acorn.distributions)

(defun beta-log-pdf (x &key (alpha 1.0d0) (beta 1.0d0))
  "Log probability density of Beta(alpha, beta).
ALPHA and BETA must be positive numbers (not AD-differentiable).
X may be an AD value (dual or tape-node).
Returns (α-1)*log(x) + (β-1)*log(1-x) - logB(α,β)."
  (assert (> (coerce alpha 'double-float) 0.0d0) nil
          "beta-log-pdf: ALPHA must be positive")
  (assert (> (coerce beta 'double-float) 0.0d0) nil
          "beta-log-pdf: BETA must be positive")
  (let ((x-real (coerce (real-value x) 'double-float)))
    (when (or (<= x-real 0.0d0) (>= x-real 1.0d0))
      (return-from beta-log-pdf +log-pdf-sentinel+)))
  (let* ((a (coerce alpha 'double-float))
         (b (coerce beta 'double-float))
         (log-beta (- (+ (log-gammaln a) (log-gammaln b))
                      (log-gammaln (+ a b)))))
    (ad:- (ad:+ (ad:* (- a 1.0d0) (ad:log x))
                (ad:* (- b 1.0d0) (ad:log (ad:- 1.0d0 x))))
          log-beta)))

(defun beta-sample (&key (alpha 1.0d0) (beta 1.0d0))
  "Sample from Beta(alpha, beta) using ratio of gamma samples.
Returns a double-float in (0, 1)."
  (assert (> alpha 0) nil "beta-sample: ALPHA must be positive")
  (assert (> beta 0) nil "beta-sample: BETA must be positive")
  (let ((x (gamma-sample :shape alpha :rate 1.0d0))
        (y (gamma-sample :shape beta :rate 1.0d0)))
    (/ x (+ x y))))
