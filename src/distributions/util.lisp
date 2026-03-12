(in-package #:cl-acorn.distributions)

;;; Lanczos approximation coefficients (g=7, n=9)
(defconstant +lanczos-g+ 7.0d0)

(defvar +lanczos-coefficients+
  (make-array 9 :element-type 'double-float
              :initial-contents
              '(0.99999999999980993d0
                676.5203681218851d0
                -1259.1392167224028d0
                771.32342877765313d0
                -176.61502916214059d0
                12.507343278686905d0
                -0.13857109526572012d0
                9.9843695780195716d-6
                1.5056327351493116d-7))
  "Lanczos coefficients — effectively constant, declared with DEFVAR
to avoid SBCL's DEFCONSTANT redefinition error on arrays.")

(defconstant +log-2pi/2+ (* 0.5d0 (log (* 2.0d0 pi)))
  "Precomputed 0.5 * log(2 * pi).")

(defun log-gammaln (z)
  "Log of the gamma function via Lanczos approximation.
Z must be a positive real number. Returns a double-float.
Used for normalization constants in gamma, beta, and poisson distributions."
  (let ((z (coerce z 'double-float)))
    (if (< z 0.5d0)
        ;; Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        (- (log (/ pi (sin (* pi z))))
           (log-gammaln (- 1.0d0 z)))
        ;; Lanczos approximation for z >= 0.5
        (let ((z (- z 1.0d0))
              (x (aref +lanczos-coefficients+ 0)))
          (loop for i from 1 below (length +lanczos-coefficients+)
                do (incf x (/ (aref +lanczos-coefficients+ i) (+ z (coerce i 'double-float)))))
          (let ((t-val (+ z +lanczos-g+ 0.5d0)))
            (+ +log-2pi/2+
               (* (+ z 0.5d0) (log t-val))
               (- t-val)
               (log x)))))))
