(in-package #:cl-acorn.distributions)

(defun poisson-log-pdf (k &key (rate 1.0d0))
  "Log probability mass of Poisson(rate).
K must be a non-negative integer. RATE may be an AD value.
Returns k*log(λ) - λ - logΓ(k+1)."
  (let ((k-float (coerce k 'double-float)))
    (ad:- (ad:* k-float (ad:log rate))
          rate
          (log-gammaln (+ k-float 1.0d0)))))

(defun poisson-sample (&key (rate 1.0d0))
  "Sample from Poisson(rate) using Knuth's algorithm.
Returns a double-float (integer value)."
  (let ((l (exp (- (coerce rate 'double-float))))
        (k 0)
        (p 1.0d0))
    (loop
      (incf k)
      (setf p (* p (random 1.0d0)))
      (when (<= p l)
        (return (coerce (1- k) 'double-float))))))
