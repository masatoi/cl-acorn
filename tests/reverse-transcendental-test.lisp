(defpackage #:cl-acorn/tests/reverse-transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

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

;;; Bug fix: log base is AD-transparent in reverse mode (M5)

(deftest test-reverse-log-wrt-base
  (testing "d/db[log_b(8)] at b=2 via gradient - base is AD-transparent"
    ;; d/db log_b(8) = d/db [ln(8)/ln(b)] = -ln(8)/(b*ln(b)^2)
    ;; at b=2: -ln(8)/(2*ln(2)^2)
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (b) (ad:log 8.0d0 b)) 2.0d0)
      (declare (ignore val))
      (let ((expected (/ (- (log 8.0d0)) (* 2.0d0 (expt (log 2.0d0) 2)))))
        (ok (approx= dx expected))))))

;;; Bug fix: expt with base=0 and tape-node power (M10)

(deftest test-reverse-expt-zero-base-tape-power
  (testing "d/dy[0^y] at y=1 returns 0 via reverse mode, not NaN"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (y) (ad:expt 0.0d0 y)) 1.0d0)
      (ok (approx= val 0.0d0))
      (ok (approx= dx 0.0d0)))))

;;; Bug fix regressions: M3, M4, M5 (reverse mode)

;;; M3: binary-expt (number, tape-node) with negative base

(deftest test-reverse-expt-negative-base-tape-power-no-type-error
  (testing "(-2)^x at x=1 via reverse mode does not signal type-error"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt -2.0d0 x)) 1.0d0)
      (ok (typep val 'double-float))
      ;; NaN check: NaN /= NaN
      (ok (= val val))
      ;; gradient is zeroed (sentinel for undefined derivative at negative base)
      (ok (approx= dx 0.0d0)))))

;;; M4: binary-expt (tape-node, number) with zero base and exponent < 1

(deftest test-reverse-expt-tape-zero-base-fractional-exponent-no-nan
  (testing "d/dx[x^0.5] at x=0 via reverse mode returns finite gradient"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x 0.5d0)) 0.0d0)
      (ok (typep val 'double-float))
      ;; NaN check
      (ok (= dx dx)))))

(deftest test-reverse-expt-tape-zero-base-exponent-one
  (testing "d/dx[x^1] at x=0 via reverse mode = 1 (correct)"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x 1.0d0)) 0.0d0)
      (ok (approx= val 0.0d0))
      (ok (approx= dx 1.0d0)))))

;;; M5: unary-log (tape-node) with non-positive value

(deftest test-reverse-log-negative-no-type-error
  (testing "log of tape-node with value -1 does not signal type-error; gradient zeroed"
    (multiple-value-bind (val dx)
        (reverse-grad-1 #'ad:log -1.0d0)
      (ok (typep val 'double-float))
      ;; gradient is zeroed sentinel (log undefined for negative input)
      (ok (approx= dx 0.0d0)))))

(deftest test-reverse-log-zero-no-type-error
  (testing "log of tape-node with value 0 does not signal type-error; gradient zeroed"
    (multiple-value-bind (val dx)
        (reverse-grad-1 #'ad:log 0.0d0)
      (ok (typep val 'double-float))
      (ok (approx= dx 0.0d0)))))

;;; Cross-validate composite function with forward-mode

(deftest test-reverse-composite-matches-forward
  (testing "reverse gradient of exp(sin(x)) matches forward derivative"
    (let ((fn (lambda (x) (ad:exp (ad:sin x)))))
      (multiple-value-bind (fwd-val fwd-grad) (ad:derivative fn 1.0d0)
        (multiple-value-bind (rev-val rev-grad) (reverse-grad-1 fn 1.0d0)
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad rev-grad)))))))
