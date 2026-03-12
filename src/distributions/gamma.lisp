(in-package #:cl-acorn.distributions)

(defun gamma-log-pdf (x &key (shape 1.0d0) (rate 1.0d0))
  "Log probability density of Gamma(shape, rate).
SHAPE must be a positive number (not AD-differentiable).
X and RATE may be AD values (dual or tape-node).
Returns (shape-1)*log(x) - rate*x + shape*log(rate) - logΓ(shape)."
  (let ((k (coerce shape 'double-float)))
    (ad:- (ad:+ (ad:* (- k 1.0d0) (ad:log x))
                (ad:* k (ad:log rate)))
          (ad:* rate x)
          (log-gammaln k))))

(defun gamma-sample (&key (shape 1.0d0) (rate 1.0d0))
  "Sample from Gamma(shape, rate) using Marsaglia-Tsang method.
Returns a double-float."
  (let ((k (coerce shape 'double-float))
        (r (coerce rate 'double-float)))
    (if (< k 1.0d0)
        ;; For shape < 1: X = Y * U^(1/shape) where Y ~ Gamma(shape+1, 1)
        (let ((u (max double-float-epsilon (random 1.0d0))))
          (/ (* (gamma-sample :shape (+ k 1.0d0) :rate 1.0d0)
                (expt u (/ 1.0d0 k)))
             r))
        ;; Marsaglia & Tsang for shape >= 1
        (let* ((d (- k (/ 1.0d0 3.0d0)))
               (c (/ 1.0d0 (sqrt (* 9.0d0 d)))))
          (/ (loop
               (let* ((x (normal-sample))
                      (v (+ 1.0d0 (* c x))))
                 (when (> v 0.0d0)
                   (let* ((v (* v v v))
                          (u (random 1.0d0)))
                     (when (or (< u (- 1.0d0 (* 0.0331d0 x x x x)))
                               (< (log u) (+ (* 0.5d0 x x)
                                             (* d (+ (- 1.0d0 v) (log v))))))
                       (return (* d v)))))))
             r)))))
