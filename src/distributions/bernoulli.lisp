(in-package #:cl-acorn.distributions)

(defun bernoulli-log-pdf (x &key (prob 0.5d0))
  "Log probability mass of Bernoulli(prob).
X must be 0 or 1 (as a number). PROB may be an AD value.
Returns x*log(p) + (1-x)*log(1-p)."
  (let ((x (coerce x 'double-float)))
    (ad:+ (ad:* x (ad:log prob))
          (ad:* (- 1.0d0 x) (ad:log (ad:- 1.0d0 prob))))))

(defun bernoulli-sample (&key (prob 0.5d0))
  "Sample from Bernoulli(prob). Returns 1.0d0 or 0.0d0."
  (if (< (random 1.0d0) (coerce prob 'double-float)) 1.0d0 0.0d0))
