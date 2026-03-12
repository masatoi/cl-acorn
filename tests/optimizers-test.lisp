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

;;; --- Adam tests ---

(deftest test-adam-state-initialization
  (testing "make-adam-state creates zero-initialized state"
    (let ((state (opt:make-adam-state 3)))
      (ok (= (length (opt:adam-state-m state)) 3))
      (ok (= (length (opt:adam-state-v state)) 3))
      (ok (every (lambda (x) (= x 0.0d0)) (opt:adam-state-m state)))
      (ok (every (lambda (x) (= x 0.0d0)) (opt:adam-state-v state)))
      (ok (= (opt:adam-state-step state) 0)))))

(deftest test-adam-step-basic
  (testing "Adam step returns updated params and increments step"
    (let ((state (opt:make-adam-state 2))
          (params '(1.0d0 2.0d0))
          (grads '(0.5d0 -1.0d0)))
      (let ((result (opt:adam-step params grads state)))
        (ok (= (length result) 2))
        (ok (= (opt:adam-state-step state) 1))
        ;; After one step, params should have changed
        (ok (not (approx= (first result) 1.0d0 1d-15)))))))

(deftest test-adam-converges-quadratic
  (testing "Adam converges to minimum of f(x,y) = (x-1)^2 + (y-2)^2"
    (let ((params '(0.0d0 0.0d0))
          (state (opt:make-adam-state 2)))
      (dotimes (i 1000)
        (let* ((x (first params))
               (y (second params))
               (grads (list (* 2.0d0 (- x 1.0d0))
                            (* 2.0d0 (- y 2.0d0)))))
          (setf params (opt:adam-step params grads state :lr 0.01d0))))
      (ok (approx= (first params) 1.0d0 0.05d0))
      (ok (approx= (second params) 2.0d0 0.05d0)))))
