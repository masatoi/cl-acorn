;;; examples/06-pid-control/main.lisp --- PID controller auto-tuning via AD
;;;
;;; Demonstrates automatic differentiation through a simulation loop:
;;; a PID controller drives a first-order plant, and gradient descent
;;; on the Integral of Squared Error (ISE) tunes Kp, Ki, Kd.
;;; The gradients dISE/dKp, dISE/dKi, dISE/dKd are computed exactly
;;; via cl-acorn's forward-mode AD, propagating derivatives through
;;; 500 discrete time steps.

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.pid-control
  (:use #:cl))

(in-package #:cl-acorn.examples.pid-control)

;;; ---------------------------------------------------------------------------
;;; Simulation parameters
;;; ---------------------------------------------------------------------------

(defconstant +dt+ 0.01d0
  "Simulation time step (seconds).")

(defconstant +t-final+ 5.0d0
  "Total simulation duration (seconds).")

(defconstant +n-steps+ 500
  "Number of simulation steps (T_final / dt).")

(defconstant +setpoint+ 1.0d0
  "Step response setpoint.")

;;; ---------------------------------------------------------------------------
;;; PID simulation with AD-compatible arithmetic
;;; ---------------------------------------------------------------------------

(defun simulate-pid (kp ki kd)
  "Simulate a PID-controlled first-order plant and return the ISE.

Plant model: G(s) = 1/(s+1), Euler discretization: dy = (-y + u) * dt.
PID law: u = Kp*e + Ki*integral(e) + Kd*(de/dt).

KP, KI, KD may be dual numbers; all internal arithmetic uses ad:
operations so that derivatives propagate through the simulation loop."
  (let ((y 0.0d0)
        (integral 0.0d0)
        (e-prev 0.0d0)
        (ise 0.0d0))
    (dotimes (i +n-steps+ ise)
      (let* ((e (ad:- +setpoint+ y))
             (de/dt (ad:/ (ad:- e e-prev) +dt+))
             (u (ad:+ (ad:* kp e)
                      (ad:* ki integral)
                      (ad:* kd de/dt)))
             (dy (ad:* (ad:+ (ad:- y) u) +dt+)))
        (setf y (ad:+ y dy))
        (setf integral (ad:+ integral (ad:* e +dt+)))
        (setf ise (ad:+ ise (ad:* (ad:* e e) +dt+)))
        (setf e-prev e)))))

;;; ---------------------------------------------------------------------------
;;; Gradient-descent optimization
;;; ---------------------------------------------------------------------------

(defun optimize-pid (&key (lr 0.5d0) (epochs 100) (kp-init 1.0d0)
                          (ki-init 0.1d0) (kd-init 0.0d0))
  "Optimize PID gains by gradient descent on the ISE cost function.

Computes dISE/dKp, dISE/dKi, dISE/dKd via separate ad:derivative
calls (one per gain), then updates each gain proportionally.

Returns (values optimized-kp optimized-ki optimized-kd final-ise)."
  (let ((kp (coerce kp-init 'double-float))
        (ki (coerce ki-init 'double-float))
        (kd (coerce kd-init 'double-float)))
    (format t "~&=== PID Auto-Tuning via AD Gradient Descent ===~%")
    (format t "Plant: G(s) = 1/(s+1), dt=~,2F, T=~,1F, setpoint=~,1F~%"
            +dt+ +t-final+ +setpoint+)
    (format t "Initial gains: Kp=~,4F  Ki=~,4F  Kd=~,4F~%"
            kp ki kd)
    (format t "Learning rate=~,4F, epochs=~D~%~%" lr epochs)
    (format t "~6A  ~12A  ~10A  ~10A  ~10A~%"
            "Epoch" "ISE" "Kp" "Ki" "Kd")
    (format t "~A~%" (make-string 52 :initial-element #\-))
    (dotimes (epoch epochs (values kp ki kd (simulate-pid kp ki kd)))
      (multiple-value-bind (ise dise/dkp)
          (ad:derivative (lambda (k) (simulate-pid k ki kd)) kp)
        (multiple-value-bind (ise2 dise/dki)
            (ad:derivative (lambda (k) (simulate-pid kp k kd)) ki)
          (declare (ignore ise2))
          (multiple-value-bind (ise3 dise/dkd)
              (ad:derivative (lambda (k) (simulate-pid kp ki k)) kd)
            (declare (ignore ise3))
            (when (zerop (mod epoch 10))
              (format t "~6D  ~12,6F  ~10,4F  ~10,4F  ~10,4F~%"
                      epoch ise kp ki kd))
            (setf kp (cl:- kp (cl:* lr dise/dkp)))
            (setf ki (cl:- ki (cl:* lr dise/dki)))
            (setf kd (cl:- kd (cl:* lr dise/dkd)))))))))

;;; ---------------------------------------------------------------------------
;;; Step response display (plain CL arithmetic)
;;; ---------------------------------------------------------------------------

(defun simulate-step-response (kp ki kd)
  "Simulate the step response and return a list of (time y) pairs.
Uses plain CL arithmetic for display purposes."
  (let ((y 0.0d0)
        (integral 0.0d0)
        (e-prev 0.0d0)
        (result '()))
    (dotimes (i +n-steps+ (nreverse result))
      (let* ((time (cl:* i +dt+))
             (e (cl:- +setpoint+ y))
             (de/dt (/ (cl:- e e-prev) +dt+))
             (u (cl:+ (cl:* kp e) (cl:* ki integral) (cl:* kd de/dt)))
             (dy (cl:* (cl:+ (cl:- y) u) +dt+)))
        (when (zerop (mod i 50))
          (push (list time y) result))
        (setf y (cl:+ y dy))
        (setf integral (cl:+ integral (cl:* e +dt+)))
        (setf e-prev e)))))

(defun print-step-response (kp ki kd label)
  "Print the step response at selected time points.
KP, KI, KD are plain numbers; uses plain CL arithmetic."
  (format t "~&~%--- ~A ---~%" label)
  (format t "Gains: Kp=~,4F  Ki=~,4F  Kd=~,4F~%~%" kp ki kd)
  (format t "~8A  ~12A  ~12A~%"
          "Time" "Output" "Error")
  (format t "~A~%" (make-string 36 :initial-element #\-))
  (dolist (pair (simulate-step-response kp ki kd))
    (let* ((time (first pair))
           (y (second pair))
           (e (cl:- +setpoint+ y)))
      (format t "~8,2F  ~12,6F  ~12,6F~%"
              time y e))))

;;; ---------------------------------------------------------------------------
;;; Main entry point
;;; ---------------------------------------------------------------------------

(defun run-examples ()
  "Demonstrate PID auto-tuning via automatic differentiation."
  ;; Show initial (untuned) step response
  (let ((kp0 1.0d0) (ki0 0.1d0) (kd0 0.0d0))
    (print-step-response kp0 ki0 kd0
                         "Initial Step Response (before tuning)")
    ;; Optimize
    (multiple-value-bind (kp ki kd final-ise)
        (optimize-pid :kp-init kp0 :ki-init ki0 :kd-init kd0)
      (format t "~%Optimized gains: Kp=~,4F  Ki=~,4F  Kd=~,4F~%"
              kp ki kd)
      (format t "Final ISE: ~,6F~%" final-ise)
      ;; Show optimized step response
      (print-step-response kp ki kd
                           "Optimized Step Response (after tuning)"))))

(run-examples)
