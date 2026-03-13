(in-package #:cl-acorn.distributions)

(defun bernoulli-log-pdf (x &key (prob 0.5d0))
  "Log probability mass of Bernoulli(prob).
X must be 0 or 1 (as a number). PROB may be an AD value.
Returns x*log(p) + (1-x)*log(1-p).
Returns +log-pdf-sentinel+ if prob is outside [0, 1] or x is not in {0, 1}.
At boundary prob=0: returns 0.0d0 for x=0, sentinel for x=1.
At boundary prob=1: returns 0.0d0 for x=1, sentinel for x=0."
  (let ((p-real (coerce (real-value prob) 'double-float))
        (x-val (coerce x 'double-float)))
    (when (and (/= x-val 0.0d0) (/= x-val 1.0d0))
      (return-from bernoulli-log-pdf +log-pdf-sentinel+))
    (when (or (< p-real 0.0d0) (> p-real 1.0d0))
      (return-from bernoulli-log-pdf +log-pdf-sentinel+))
    (when (= p-real 0.0d0)
      (return-from bernoulli-log-pdf (if (= x-val 0.0d0) 0.0d0 +log-pdf-sentinel+)))
    (when (= p-real 1.0d0)
      (return-from bernoulli-log-pdf (if (= x-val 1.0d0) 0.0d0 +log-pdf-sentinel+))))
  (let ((x (coerce x 'double-float)))
    (ad:+ (ad:* x (ad:log prob))
          (ad:* (- 1.0d0 x) (ad:log (ad:- 1.0d0 prob))))))

(defun bernoulli-sample (&key (prob 0.5d0))
  "Sample from Bernoulli(prob). Returns 1.0d0 or 0.0d0."
  (assert (and (>= prob 0) (<= prob 1)) nil
          "bernoulli-sample: PROB must be in [0, 1]")
  (if (< (random 1.0d0) (coerce prob 'double-float)) 1.0d0 0.0d0))
