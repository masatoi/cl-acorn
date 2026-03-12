(defpackage #:cl-acorn/tests/reverse-transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/reverse-transcendental-test)

;;; Helper
(defun reverse-grad-1 (fn x-val)
  (let ((cl-acorn.ad::*tape* (list t)))
    (let* ((x (cl-acorn.ad::make-node (coerce x-val 'double-float) nil))
           (result (funcall fn x)))
      (setf cl-acorn.ad::*tape* (butlast cl-acorn.ad::*tape*))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient x)))))

;;; Cross-validate each function: reverse gradient must match forward derivative

(deftest test-reverse-sin
  (testing "d/dx[sin(x)] = cos(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:sin 1.0d0)
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= dx (cos 1.0d0))))))

(deftest test-reverse-cos
  (testing "d/dx[cos(x)] = -sin(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:cos 1.0d0)
      (ok (approx= val (cos 1.0d0)))
      (ok (approx= dx (- (sin 1.0d0)))))))

(deftest test-reverse-tan
  (testing "d/dx[tan(x)] = 1/cos^2(x) at x=0.5"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:tan 0.5d0)
      (ok (approx= val (tan 0.5d0)))
      (ok (approx= dx (/ 1.0d0 (expt (cos 0.5d0) 2)))))))

(deftest test-reverse-exp
  (testing "d/dx[exp(x)] = exp(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:exp 1.0d0)
      (ok (approx= val (exp 1.0d0)))
      (ok (approx= dx (exp 1.0d0))))))

(deftest test-reverse-log
  (testing "d/dx[log(x)] = 1/x at x=2"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:log 2.0d0)
      (ok (approx= val (log 2.0d0)))
      (ok (approx= dx 0.5d0)))))

(deftest test-reverse-sqrt
  (testing "d/dx[sqrt(x)] = 1/(2*sqrt(x)) at x=4"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:sqrt 4.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= dx 0.25d0)))))

(deftest test-reverse-abs
  (testing "d/dx[abs(x)] = signum(x) at x=-3"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:abs -3.0d0)
      (ok (approx= val 3.0d0))
      (ok (approx= dx -1.0d0)))))

(deftest test-reverse-expt-tape-number
  (testing "d/dx[x^3] = 3x^2 at x=2"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x 3)) 2.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= dx 12.0d0)))))

(deftest test-reverse-expt-number-tape
  (testing "d/dx[2^x] = 2^x * ln(2) at x=3"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt 2 x)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= dx (* 8.0d0 (log 2.0d0)))))))

(deftest test-reverse-expt-tape-tape
  (testing "d/dx[x^x] at x=2 (uses exp(x*ln(x)) decomposition)"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x x)) 2.0d0)
      (ok (approx= val 4.0d0))
      (ok (approx= dx (* 4.0d0 (+ 1.0d0 (log 2.0d0))))))))

(deftest test-reverse-log-with-base
  (testing "d/dx[log_10(x)] = 1/(x * ln(10)) at x=100"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:log x 10)) 100.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= dx (/ 1.0d0 (* 100.0d0 (log 10.0d0))))))))

;;; Cross-validate composite function with forward-mode

(deftest test-reverse-composite-matches-forward
  (testing "reverse gradient of exp(sin(x)) matches forward derivative"
    (let ((fn (lambda (x) (ad:exp (ad:sin x)))))
      (multiple-value-bind (fwd-val fwd-grad) (ad:derivative fn 1.0d0)
        (multiple-value-bind (rev-val rev-grad) (reverse-grad-1 fn 1.0d0)
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad rev-grad)))))))
