;;; examples/04-sensitivity/main.lisp — Parameter sensitivity analysis via AD
;;;
;;; Demonstrates using cl-acorn's automatic differentiation for parameter
;;; sensitivity analysis: given a model f(p), compute df/dp exactly to
;;; understand how sensitive the output is to each parameter.

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.sensitivity
  (:use #:cl))

(in-package #:cl-acorn.examples.sensitivity)

;;; ---------------------------------------------------------------------------
;;; Model 1: Simple Pendulum Period
;;;
;;; T(L) = 2*pi*sqrt(L/g)   (small-angle approximation)
;;;
;;; Analytical derivative:  dT/dL = pi / sqrt(L*g)
;;; ---------------------------------------------------------------------------

(defconstant +g+ 9.80665d0
  "Standard gravitational acceleration (m/s^2).")

(defun pendulum-period (length)
  "Period of a simple pendulum as a function of LENGTH.
Uses cl-acorn.ad arithmetic so LENGTH may be a dual number."
  (ad:* (* 2.0d0 pi) (ad:sqrt (ad:/ length +g+))))

(defun analytical-dt/dl (length)
  "Analytical derivative dT/dL = pi / sqrt(L * g)."
  (/ pi (sqrt (* length +g+))))

(defun run-pendulum-sensitivity ()
  "Print a table comparing AD and analytical dT/dL for various pendulum lengths."
  (format t "~&=== Model 1: Simple Pendulum Period T(L) = 2*pi*sqrt(L/g) ===~%")
  (format t "g = ~,5F m/s^2~%~%" +g+)
  (format t "~8A  ~12A  ~18A  ~18A  ~12A~%"
          "L (m)" "T (s)" "dT/dL (AD)" "dT/dL (exact)" "error")
  (format t "~A~%" (make-string 72 :initial-element #\-))
  (dolist (length '(0.25d0 0.5d0 1.0d0 1.5d0 2.0d0 3.0d0 5.0d0))
    (multiple-value-bind (period dt/dl)
        (ad:derivative #'pendulum-period length)
      (let* ((exact (analytical-dt/dl length))
             (err (abs (- dt/dl exact))))
        (format t "~8,2F  ~12,6F  ~18,12F  ~18,12F  ~12,2E~%"
                length period dt/dl exact err)))))

;;; ---------------------------------------------------------------------------
;;; Model 2: Damped Oscillation
;;;
;;; x(t; gamma) = A * exp(-gamma * t) * cos(omega * t)
;;;
;;; Sensitivity to damping parameter gamma:
;;;   dx/dgamma = -t * A * exp(-gamma * t) * cos(omega * t)
;;; ---------------------------------------------------------------------------

(defconstant +a+ 1.0d0
  "Initial amplitude.")

(defconstant +omega+ (* 2.0d0 pi)
  "Angular frequency (rad/s).")

(defun damped-oscillation (gamma time)
  "Position of a damped oscillator at TIME for damping coefficient GAMMA.
GAMMA may be a dual number for AD sensitivity analysis."
  (ad:* +a+ (ad:* (ad:exp (ad:* (ad:- gamma) time))
                   (cl:cos (* +omega+ time)))))

(defun analytical-dx/dgamma (gamma time)
  "Analytical derivative dx/dgamma = -t * A * exp(-gamma*t) * cos(omega*t)."
  (* (- time) +a+ (exp (* (- gamma) time)) (cos (* +omega+ time))))

(defun run-damped-sensitivity ()
  "Print a table comparing AD and analytical dx/dgamma at various times."
  (let ((gamma 0.5d0))
    (format t "~&=== Model 2: Damped Oscillation x(t;gamma) = A*exp(-gamma*t)*cos(omega*t) ===~%")
    (format t "A = ~,1F, omega = 2*pi, gamma = ~,1F~%~%" +a+ gamma)
    (format t "~8A  ~12A  ~18A  ~18A  ~12A~%"
            "t (s)" "x(t)" "dx/dgamma (AD)" "dx/dgamma (exact)" "error")
    (format t "~A~%" (make-string 72 :initial-element #\-))
    (dolist (time '(0.1d0 0.25d0 0.5d0 1.0d0 1.5d0 2.0d0 3.0d0))
      (multiple-value-bind (x dx/dgamma)
          (ad:derivative (lambda (g) (damped-oscillation g time)) gamma)
        (let* ((exact (analytical-dx/dgamma gamma time))
               (err (abs (- dx/dgamma exact))))
          (format t "~8,2F  ~12,6F  ~18,12F  ~18,12F  ~12,2E~%"
                  time x dx/dgamma exact err))))))

;;; ---------------------------------------------------------------------------
;;; Main entry point
;;; ---------------------------------------------------------------------------

(defun run-examples ()
  "Run all parameter sensitivity demonstrations."
  (run-pendulum-sensitivity)
  (terpri)
  (run-damped-sensitivity))

(run-examples)
