;;; examples/03-newton-method/main.lisp — Newton-Raphson root finding via AD
;;;
;;; Demonstrates using cl-acorn's automatic differentiation to compute
;;; exact derivatives for Newton's method, eliminating the need for
;;; hand-derived or finite-difference approximations.

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.newton-method
  (:use #:cl))

(in-package #:cl-acorn.examples.newton-method)

(defun newton-raphson (f x0 &key (tolerance 1d-12) (max-iter 50))
  "Find a root of F near X0 using Newton-Raphson iteration with AD.
F must be a function of one argument using cl-acorn.ad arithmetic.
Returns (values root iterations)."
  (format t "~&~3A  ~22A  ~22A  ~22A~%" "n" "x_n" "f(x_n)" "f'(x_n)")
  (format t "~A~%" (make-string 73 :initial-element #\-))
  (loop :with x = (coerce x0 'double-float)
        :for n :from 0 :below max-iter
        :do (multiple-value-bind (fx fpx)
                (ad:derivative f x)
              (format t "~3D  ~22,15E  ~22,15E  ~22,15E~%" n x fx fpx)
              (when (< (abs fx) tolerance)
                (return (values x n)))
              (when (zerop fpx)
                (error "Zero derivative at x = ~A; Newton step undefined." x))
              (setf x (- x (/ fx fpx))))
        :finally (warn "Did not converge within ~D iterations." max-iter)))

(defun run-examples ()
  "Run Newton-Raphson on two classic problems."
  ;; Problem 1: x^3 - 2x - 5 = 0, x0 = 2.0
  (format t "~&=== Problem 1: x^3 - 2x - 5 = 0 ===~%")
  (format t "Starting from x0 = 2.0~%~%")
  (multiple-value-bind (root iters)
      (newton-raphson (lambda (x) (ad:- (ad:- (ad:expt x 3) (ad:* 2 x)) 5))
                      2.0d0)
    (format t "~%Root:       ~,15F~%" root)
    (format t "Iterations: ~D~%" iters)
    (let ((verify (- (expt root 3) (* 2 root) 5)))
      (format t "Verify:     f(root) = ~E~%" verify)))

  (terpri)

  ;; Problem 2: cos(x) - x = 0, x0 = 1.0
  (format t "~&=== Problem 2: cos(x) - x = 0 (Dottie number) ===~%")
  (format t "Starting from x0 = 1.0~%~%")
  (multiple-value-bind (root iters)
      (newton-raphson (lambda (x) (ad:- (ad:cos x) x))
                      1.0d0)
    (format t "~%Root:       ~,15F~%" root)
    (format t "Iterations: ~D~%" iters)
    (let ((verify (- (cos root) root)))
      (format t "Verify:     f(root) = ~E~%" verify))))

(run-examples)
