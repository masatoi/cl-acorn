;;;; main.lisp --- Bayesian linear regression via Variational Inference (ADVI)
;;;;
;;;; Demonstrates using cl-acorn's mean-field ADVI to approximate the posterior
;;;; distribution of w and b in the model y = w*x + b (sigma=0.5).
;;;; VI is faster than MCMC but returns a Gaussian approximation, not exact samples.
;;;;
;;;; Usage:
;;;;   (load "examples/11-vi-bayes-regression/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.vi-bayes-regression
  (:use #:cl)
  (:export #:log-posterior
           #:run-example))

(in-package #:cl-acorn.examples.vi-bayes-regression)

;;; --------------------------------------------------------------------------
;;; Synthetic data: y = 2x + 1 + noise, noise ~ N(0, 0.5^2)
;;; --------------------------------------------------------------------------

(defparameter *data-xs*
  #(0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0 0.9d0 1.0d0
    1.1d0 1.2d0 1.3d0 1.4d0 1.5d0 1.6d0 1.7d0 1.8d0 1.9d0 2.0d0))

(defparameter *data-ys*
  #(1.35d0 1.52d0 1.58d0 1.84d0 2.12d0 2.31d0 2.45d0 2.87d0 2.91d0 3.08d0
    3.22d0 3.44d0 3.58d0 3.87d0 4.01d0 4.28d0 4.38d0 4.67d0 4.81d0 5.12d0))

;;; --------------------------------------------------------------------------
;;; Model: log p(w, b | data)
;;; --------------------------------------------------------------------------

(defun log-posterior (params)
  "Log-posterior for y = w*x + b, sigma=0.5.
PARAMS is a list (w b). Uses AD-transparent arithmetic for gradient computation."
  (let* ((w (first params))
         (b (second params))
         (sigma 0.5d0)
         (log-prior (ad:+ (dist:normal-log-pdf w :mu 0.0d0 :sigma 1.0d0)
                          (dist:normal-log-pdf b :mu 0.0d0 :sigma 10.0d0)))
         (log-lik 0.0d0))
    (dotimes (i (length *data-xs*))
      (let ((pred (ad:+ (ad:* w (aref *data-xs* i)) b)))
        (setf log-lik (ad:+ log-lik
                            (dist:normal-log-pdf pred
                                                 :mu (aref *data-ys* i)
                                                 :sigma sigma)))))
    (ad:+ log-prior log-lik)))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-example ()
  "Run VI (ADVI) on the Bayesian linear regression model and print results."
  (format t "~%=== VI (ADVI) Bayesian Linear Regression ===~%")
  (format t "Model: y = w*x + b,  sigma=0.5~%")
  (format t "True values: w=2.0, b=1.0~%")
  (format t "Data: ~D observations~%~%" (length *data-xs*))
  (format t "Running VI (2000 iterations)...~%")
  (multiple-value-bind (mu-list sigma-list elbo-history)
      (infer:vi #'log-posterior 2
                :n-iterations 2000
                :n-elbo-samples 10
                :lr 0.01d0)
    (format t "~%Variational posterior (mean-field Gaussian):~%")
    (format t "  w:  mean=~6,3F  std=~6,3F   [true: 2.0]~%"
            (first mu-list) (first sigma-list))
    (format t "  b:  mean=~6,3F  std=~6,3F   [true: 1.0]~%"
            (second mu-list) (second sigma-list))
    (format t "Final ELBO: ~,3F  (higher is better)~%"
            (car (last elbo-history)))))

(run-example)
