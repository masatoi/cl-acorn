;;;; main.lisp --- Linear regression via gradient descent with automatic differentiation
;;;;
;;;; Demonstrates using cl-acorn's forward-mode AD to optimize a linear model
;;;; y = w*x + b on the Iris sepal-length vs sepal-width dataset.
;;;;
;;;; Usage:
;;;;   (load "examples/01-curve-fitting/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.curve-fitting
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad))
  (:export #:mse-loss
           #:gradient-descent
           #:ols-solution
           #:run-examples))

(in-package #:cl-acorn.examples.curve-fitting)

(load (merge-pathnames "data.lisp" *load-pathname*))

;;; --------------------------------------------------------------------------
;;; Loss function
;;; --------------------------------------------------------------------------

(defun mse-loss (w b xs ys)
  "Mean squared error for a linear model y = w*x + b.
W and B may be dual numbers (for AD) or plain numbers.
XS and YS are simple-vectors of double-float observations."
  (let ((n (length xs))
        (total 0.0d0))
    (loop for i below n
          do (let ((residual (ad:- (ad:+ (ad:* w (aref xs i)) b)
                                   (aref ys i))))
               (setf total (ad:+ total (ad:* residual residual)))))
    (ad:/ total n)))

;;; --------------------------------------------------------------------------
;;; Gradient descent
;;; --------------------------------------------------------------------------

(defun gradient-descent (xs ys &key (lr 0.01d0) (epochs 200))
  "Train a linear model y = w*x + b using gradient descent with AD.
For each parameter, creates a closure over the other parameter and uses
AD:DERIVATIVE to compute the partial derivative of the MSE loss.
Returns (values w b final-loss)."
  (let ((w 0.0d0)
        (b 0.0d0))
    (dotimes (epoch epochs)
      ;; Partial derivative w.r.t. w (holding b fixed)
      (multiple-value-bind (loss dw)
          (ad:derivative (lambda (w) (mse-loss w b xs ys)) w)
        ;; Partial derivative w.r.t. b (holding w fixed)
        (multiple-value-bind (loss-b db)
            (ad:derivative (lambda (b) (mse-loss w b xs ys)) b)
          (declare (ignore loss-b))
          (when (zerop (mod epoch 20))
            (format t "  Epoch ~3D  |  Loss ~8,4F  |  w ~8,4F  |  b ~8,4F~%"
                    epoch loss w b))
          ;; Update parameters
          (setf w (- w (* lr dw)))
          (setf b (- b (* lr db))))))
    (values w b (mse-loss w b xs ys))))

;;; --------------------------------------------------------------------------
;;; Closed-form OLS solution
;;; --------------------------------------------------------------------------

(defun ols-solution (xs ys)
  "Ordinary least squares closed-form solution for y = w*x + b.
Returns (values w b loss)."
  (let* ((n (length xs))
         (x-mean (/ (loop for i below n sum (aref xs i)) n))
         (y-mean (/ (loop for i below n sum (aref ys i)) n))
         (numerator 0.0d0)
         (denominator 0.0d0))
    (loop for i below n
          do (let ((xd (- (aref xs i) x-mean)))
               (incf numerator (* xd (- (aref ys i) y-mean)))
               (incf denominator (* xd xd))))
    (let* ((w (/ numerator denominator))
           (b (- y-mean (* w x-mean))))
      (values w b (mse-loss w b xs ys)))))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-examples ()
  "Run gradient descent and compare with the closed-form OLS solution."
  (format t "~%========================================~%")
  (format t " Linear Regression with AD (Iris Data)~%")
  (format t "========================================~%~%")
  (format t "Model: y = w*x + b  (sepal-width vs sepal-length)~%")
  (format t "Data:  150 Iris observations~%~%")

  ;; Gradient descent
  (format t "--- Gradient Descent (lr=0.01, epochs=200) ---~%~%")
  (multiple-value-bind (gd-w gd-b gd-loss)
      (gradient-descent *iris-x* *iris-y*)

    ;; OLS
    (format t "~%--- OLS Closed-Form Solution ---~%~%")
    (multiple-value-bind (ols-w ols-b ols-loss)
        (ols-solution *iris-x* *iris-y*)

      ;; Comparison table
      (format t "~%--- Comparison ---~%~%")
      (format t "  Method  |     w      |     b      |   MSE Loss~%")
      (format t "  --------+------------+------------+-----------~%")
      (format t "  GD      | ~10,6F | ~10,6F | ~10,6F~%"
              gd-w gd-b gd-loss)
      (format t "  OLS     | ~10,6F | ~10,6F | ~10,6F~%"
              ols-w ols-b ols-loss)
      (format t "~%Note: GD with lr=0.01 and 200 epochs may not fully converge.~%")
      (format t "The AD-computed gradients are exact; convergence depends on~%")
      (format t "learning rate and iteration count.~%"))))

(run-examples)
