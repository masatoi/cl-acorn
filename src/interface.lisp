(in-package #:cl-acorn.ad)

(defun derivative (fn x)
  "Compute f(x) and f'(x) using forward-mode automatic differentiation.
Returns (values f(x) f'(x)) as double-floats.
FN must be a function of one argument using cl-acorn.ad arithmetic."
  (let* ((x-dual (make-dual x 1.0d0))
         (result (funcall fn x-dual)))
    (etypecase result
      (dual   (values (dual-real result) (dual-epsilon result)))
      (number (values (coerce result 'double-float) 0.0d0)))))
