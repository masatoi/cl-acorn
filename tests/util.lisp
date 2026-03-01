(defpackage #:cl-acorn/tests/util
  (:use #:cl)
  (:export #:approx=))

(in-package #:cl-acorn/tests/util)

(defun approx= (a b &optional (tolerance 1d-10))
  "Check if two numbers are approximately equal within tolerance."
  (< (abs (- a b)) tolerance))
