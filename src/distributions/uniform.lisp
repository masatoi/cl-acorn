(in-package #:cl-acorn.distributions)

(defun real-value (x)
  "Extract plain numeric value from X (dual, tape-node, or number)."
  (typecase x
    (ad:dual (ad:dual-real x))
    (ad:tape-node (let ((v (ad:node-value x)))
                    (if (typep v 'ad:dual) (ad:dual-real v) v)))
    (t x)))

(defun uniform-log-pdf (x &key (low 0.0d0) (high 1.0d0))
  "Log probability density of Uniform(low, high).
X may be an AD value. LOW and HIGH may be AD values.
Returns -log(high - low) if low <= x <= high, else most-negative-double-float."
  (let ((x-real (coerce (real-value x) 'double-float))
        (lo-real (coerce (real-value low) 'double-float))
        (hi-real (coerce (real-value high) 'double-float)))
    (if (and (>= x-real lo-real) (<= x-real hi-real))
        (ad:- (ad:log (ad:- high low)))
        most-negative-double-float)))

(defun uniform-sample (&key (low 0.0d0) (high 1.0d0))
  "Sample from Uniform(low, high). Returns a double-float."
  (let ((lo (coerce low 'double-float))
        (hi (coerce high 'double-float)))
    (+ lo (* (random 1.0d0) (- hi lo)))))
