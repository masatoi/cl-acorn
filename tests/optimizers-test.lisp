(defpackage #:cl-acorn/tests/optimizers-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/optimizers-test)

;;; --- SGD tests ---

(deftest test-sgd-step-basic
  (testing "SGD step updates params correctly: p - lr*g"
    (let ((result (opt:sgd-step '(1.0d0 2.0d0) '(0.5d0 -1.0d0) :lr 0.1d0)))
      ;; p1 = 1.0 - 0.1*0.5 = 0.95
      ;; p2 = 2.0 - 0.1*(-1.0) = 2.1
      (ok (approx= (first result) 0.95d0))
      (ok (approx= (second result) 2.1d0)))))

(deftest test-sgd-converges-quadratic
  (testing "SGD converges to minimum of f(x) = (x-3)^2"
    ;; grad f(x) = 2*(x-3)
    (let ((params '(0.0d0)))
      (dotimes (i 200)
        (let* ((x (first params))
               (grad (list (* 2.0d0 (- x 3.0d0)))))
          (setf params (opt:sgd-step params grad :lr 0.05d0))))
      (ok (approx= (first params) 3.0d0 0.1d0)))))
