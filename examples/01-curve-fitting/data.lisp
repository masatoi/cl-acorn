;;;; data.lisp --- Iris sepal-length vs sepal-width dataset
;;;;
;;;; Source: UCI Machine Learning Repository - Iris dataset
;;;; (Fisher, 1936; Anderson, 1935)
;;;; 150 observations: sepal-length (x) vs sepal-width (y)

(in-package #:cl-acorn.examples.curve-fitting)

(defparameter *iris-x*
  (make-array 150
              :element-type 'double-float
              :initial-contents
              '(5.1d0 4.9d0 4.7d0 4.6d0 5.0d0 5.4d0 4.6d0 5.0d0 4.4d0 4.9d0
                5.4d0 4.8d0 4.8d0 4.3d0 5.8d0 5.7d0 5.4d0 5.1d0 5.7d0 5.1d0
                5.4d0 5.1d0 4.6d0 5.1d0 4.8d0 5.0d0 5.0d0 5.2d0 5.2d0 4.7d0
                4.8d0 5.4d0 5.2d0 5.5d0 4.9d0 5.0d0 5.5d0 4.9d0 4.4d0 5.1d0
                5.0d0 4.5d0 4.4d0 5.0d0 5.1d0 4.8d0 5.1d0 4.6d0 5.3d0 5.0d0
                7.0d0 6.4d0 6.9d0 5.5d0 6.5d0 5.7d0 6.3d0 4.9d0 6.6d0 5.2d0
                5.0d0 5.9d0 6.0d0 6.1d0 5.6d0 6.7d0 5.6d0 5.8d0 6.2d0 5.6d0
                5.9d0 6.1d0 6.3d0 6.1d0 6.4d0 6.6d0 6.8d0 6.7d0 6.0d0 5.7d0
                5.5d0 5.5d0 5.8d0 6.0d0 5.4d0 6.0d0 6.7d0 6.3d0 5.6d0 5.5d0
                5.5d0 6.1d0 5.8d0 5.0d0 5.6d0 5.7d0 5.7d0 6.2d0 5.1d0 5.7d0
                6.3d0 5.8d0 7.1d0 6.3d0 6.5d0 7.6d0 4.9d0 7.3d0 6.7d0 7.2d0
                6.5d0 6.4d0 6.8d0 5.7d0 5.8d0 6.4d0 6.5d0 7.7d0 7.7d0 6.0d0
                6.9d0 5.6d0 7.7d0 6.3d0 6.7d0 7.2d0 6.2d0 6.1d0 6.4d0 7.2d0
                7.4d0 7.9d0 6.4d0 6.3d0 6.1d0 7.7d0 6.3d0 6.4d0 6.0d0 6.9d0
                6.7d0 6.9d0 5.8d0 6.8d0 6.7d0 6.7d0 6.3d0 6.5d0 6.2d0 5.9d0))
  "Iris sepal-length in cm (150 observations).")

(defparameter *iris-y*
  (make-array 150
              :element-type 'double-float
              :initial-contents
              '(3.5d0 3.0d0 3.2d0 3.1d0 3.6d0 3.9d0 3.4d0 3.4d0 2.9d0 3.1d0
                3.7d0 3.4d0 3.0d0 3.0d0 4.0d0 4.4d0 3.9d0 3.5d0 3.8d0 3.8d0
                3.4d0 3.7d0 3.6d0 3.3d0 3.4d0 3.0d0 3.4d0 3.5d0 3.4d0 3.2d0
                3.1d0 3.4d0 4.1d0 4.2d0 3.1d0 3.2d0 3.5d0 3.6d0 3.0d0 3.4d0
                3.5d0 2.3d0 3.2d0 3.5d0 3.8d0 3.0d0 3.8d0 3.2d0 3.7d0 3.3d0
                3.2d0 3.2d0 3.1d0 2.3d0 2.8d0 2.8d0 3.3d0 2.4d0 2.9d0 2.7d0
                2.0d0 3.0d0 2.2d0 2.9d0 2.9d0 3.1d0 3.0d0 2.7d0 2.2d0 2.5d0
                3.2d0 2.8d0 2.5d0 2.8d0 2.9d0 3.0d0 2.8d0 3.0d0 2.9d0 2.6d0
                2.4d0 2.4d0 2.7d0 2.7d0 3.0d0 3.4d0 3.1d0 2.3d0 3.0d0 2.5d0
                2.6d0 3.0d0 2.6d0 2.3d0 2.7d0 3.0d0 2.9d0 2.9d0 2.5d0 2.8d0
                3.3d0 2.7d0 3.0d0 2.9d0 3.0d0 3.0d0 2.5d0 2.9d0 2.5d0 3.6d0
                3.2d0 2.7d0 3.0d0 2.5d0 2.8d0 3.2d0 3.0d0 3.8d0 2.6d0 2.2d0
                3.2d0 2.8d0 2.8d0 2.7d0 3.3d0 3.2d0 2.8d0 3.0d0 2.8d0 3.0d0
                2.8d0 3.8d0 2.8d0 2.8d0 2.6d0 3.0d0 3.4d0 3.1d0 3.0d0 3.1d0
                3.1d0 3.1d0 2.7d0 3.2d0 3.3d0 3.0d0 2.5d0 3.0d0 3.4d0 3.0d0))
  "Iris sepal-width in cm (150 observations).")
