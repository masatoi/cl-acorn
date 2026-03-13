(in-package #:cl-acorn.distributions)

(defun poisson-log-pdf (k &key (rate 1.0d0))
  "Log probability mass of Poisson(rate).
K must be a non-negative integer. RATE may be an AD value.
Returns k*log(λ) - λ - logΓ(k+1).
Signals an error if k is not a non-negative integer.
Returns +log-pdf-sentinel+ if rate <= 0."
  (let ((k-int (cond ((integerp k) k)
                     ((and (realp k) (= k (floor k))) (floor k))
                     (t nil))))
    (assert (and k-int (>= k-int 0)) nil
            "poisson-log-pdf: K must be a non-negative integer value"))
  (when (<= (coerce (real-value rate) 'double-float) 0.0d0)
    (return-from poisson-log-pdf +log-pdf-sentinel+))
  (let ((k-float (coerce k 'double-float)))
    (ad:- (ad:* k-float (ad:log rate))
          rate
          (log-gammaln (+ k-float 1.0d0)))))

(defun poisson-sample (&key (rate 1.0d0))
  "Sample from Poisson(rate) using Knuth's algorithm.
Returns a double-float (integer value).
RATE must be positive and at most 700 (Knuth's algorithm underflows for larger values)."
  (assert (> rate 0) nil "poisson-sample: RATE must be positive")
  (assert (<= rate 700) nil
          "poisson-sample: RATE must be <= 700 (Knuth algorithm underflows for larger values)")
  (let ((l (exp (- (coerce rate 'double-float))))
        (k 0)
        (p 1.0d0))
    (loop
      (incf k)
      (setf p (* p (random 1.0d0)))
      (when (<= p l)
        (return (coerce (1- k) 'double-float))))))
