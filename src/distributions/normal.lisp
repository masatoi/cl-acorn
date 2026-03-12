(in-package #:cl-acorn.distributions)

(defun normal-log-pdf (x &key (mu 0.0d0) (sigma 1.0d0))
  "Log probability density of Normal(mu, sigma).
X, MU, and SIGMA may be AD values (dual or tape-node).
Returns log N(x | mu, sigma) = -0.5*log(2pi) - log(sigma) - 0.5*((x-mu)/sigma)^2."
  (let ((z (ad:/ (ad:- x mu) sigma)))
    (ad:- (ad:* -0.5d0 (ad:* z z))
          (ad:log sigma)
          +log-2pi/2+)))

(defun normal-sample (&key (mu 0.0d0) (sigma 1.0d0))
  "Sample from Normal(mu, sigma) using Box-Muller transform.
Returns a double-float."
  (let* ((mu (coerce mu 'double-float))
         (sigma (coerce sigma 'double-float))
         (u1 (max double-float-epsilon (random 1.0d0)))
         (u2 (random 1.0d0))
         (z (* (sqrt (* -2.0d0 (log u1)))
               (cos (* 2.0d0 pi u2)))))
    (+ mu (* sigma z))))
