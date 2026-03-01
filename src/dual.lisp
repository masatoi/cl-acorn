(in-package #:cl-acorn.ad)

(defclass dual ()
  ((real-part :initarg :real
              :reader dual-real
              :type double-float
              :documentation "The real (primal) component of the dual number.")
   (epsilon   :initarg :epsilon
              :reader dual-epsilon
              :type double-float
              :initform 0.0d0
              :documentation "The infinitesimal (tangent) component."))
  (:documentation "Dual number a + b*epsilon where epsilon^2 = 0.
Used for forward-mode automatic differentiation."))

(defun make-dual (real &optional (epsilon 0.0d0))
  "Construct a dual number, coercing inputs to double-float."
  (make-instance 'dual
                 :real (coerce real 'double-float)
                 :epsilon (coerce epsilon 'double-float)))

(defmethod print-object ((d dual) stream)
  (print-unreadable-object (d stream :type t)
    (format stream "~A + ~Aε" (dual-real d) (dual-epsilon d))))
