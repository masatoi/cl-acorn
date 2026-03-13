(in-package #:cl-acorn.distributions)

(defun uniform-log-pdf (x &key (low 0.0d0) (high 1.0d0))
  "Log probability density of Uniform(low, high).
X may be an AD value. LOW and HIGH may be AD values.
Returns -log(high - low) if low <= x <= high, else +log-pdf-sentinel+.
Signals an error if low >= high."
  (let ((x-real (coerce (real-value x) 'double-float))
        (lo-real (coerce (real-value low) 'double-float))
        (hi-real (coerce (real-value high) 'double-float)))
    (assert (< lo-real hi-real) nil
            "uniform-log-pdf: LOW must be less than HIGH")
    (if (and (>= x-real lo-real) (<= x-real hi-real))
        (ad:- (ad:log (ad:- high low)))
        +log-pdf-sentinel+)))

(defun uniform-sample (&key (low 0.0d0) (high 1.0d0))
  "Sample from Uniform(low, high). Returns a double-float."
  (assert (< low high) nil "uniform-sample: LOW must be less than HIGH")
  (let ((lo (coerce low 'double-float))
        (hi (coerce high 'double-float)))
    (+ lo (* (random 1.0d0) (- hi lo)))))
