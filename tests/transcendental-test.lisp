(defpackage #:cl-acorn/tests/transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/transcendental-test)

;;; sin

(deftest test-sin-dual
  (testing "sin(1+1e) = sin(1) + cos(1)*e"
    (let ((result (ad:sin (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (sin 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (cos 1.0d0))))))

(deftest test-sin-number
  (testing "sin delegates to cl:sin for numbers"
    (ok (= (ad:sin 0.0d0) (sin 0.0d0)))))

;;; cos

(deftest test-cos-dual
  (testing "cos(1+1e) = cos(1) + -sin(1)*e"
    (let ((result (ad:cos (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (cos 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (- (sin 1.0d0)))))))

(deftest test-cos-number
  (testing "cos delegates to cl:cos for numbers"
    (ok (= (ad:cos 0.0d0) (cos 0.0d0)))))

;;; tan

(deftest test-tan-dual
  (testing "tan(1+1e) = tan(1) + 1/cos^2(1)*e"
    (let ((result (ad:tan (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (tan 1.0d0)))
      (ok (approx= (ad:dual-epsilon result)
                    (/ 1.0d0 (expt (cos 1.0d0) 2)))))))

;;; exp

(deftest test-exp-dual
  (testing "exp(1+1e) = exp(1) + exp(1)*e"
    (let ((result (ad:exp (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (exp 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (exp 1.0d0))))))

(deftest test-exp-number
  (testing "exp delegates to cl:exp for numbers"
    (ok (= (ad:exp 1.0d0) (exp 1.0d0)))))

;;; log

(deftest test-log-dual
  (testing "log(2+1e) = log(2) + 1/2*e"
    (let ((result (ad:log (ad:make-dual 2.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (log 2.0d0)))
      (ok (approx= (ad:dual-epsilon result) 0.5d0)))))

(deftest test-log-number
  (testing "log delegates to cl:log for numbers"
    (ok (= (ad:log 2.0d0) (log 2.0d0)))))

(deftest test-log-with-base
  (testing "log with base argument"
    (let ((result (ad:log (ad:make-dual 100.0d0 1.0d0) 10.0d0)))
      (ok (approx= (ad:dual-real result) (log 100.0d0 10.0d0)))
      ;; d/dx log_b(x) = 1/(x*ln(b))
      (ok (approx= (ad:dual-epsilon result)
                    (/ 1.0d0 (* 100.0d0 (log 10.0d0))))))))

;;; sqrt

(deftest test-sqrt-dual
  (testing "sqrt(4+1e) = sqrt(4) + 1/(2*sqrt(4))*e = 2 + 0.25e"
    (let ((result (ad:sqrt (ad:make-dual 4.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) 2.0d0))
      (ok (approx= (ad:dual-epsilon result) 0.25d0)))))

(deftest test-sqrt-number
  (testing "sqrt delegates to cl:sqrt for numbers"
    (ok (= (ad:sqrt 4.0d0) (sqrt 4.0d0)))))

;;; abs

(deftest test-abs-dual-positive
  (testing "abs of positive dual"
    (let ((result (ad:abs (ad:make-dual 3.0d0 2.0d0))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-abs-dual-negative
  (testing "abs of negative dual"
    (let ((result (ad:abs (ad:make-dual -3.0d0 2.0d0))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) -2.0d0)))))

(deftest test-abs-number
  (testing "abs delegates to cl:abs for numbers"
    (ok (= (ad:abs -5) 5))))

;;; expt

(deftest test-expt-dual-number
  (testing "expt((2+1e), 3) = 8 + 12e (power rule: 3*2^2*1)"
    (let ((result (ad:expt (ad:make-dual 2.0d0 1.0d0) 3)))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-expt-number-dual
  (testing "expt(2, (3+1e)) = 2^3 + 2^3*ln(2)*e"
    (let ((result (ad:expt 2 (ad:make-dual 3.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) (* 8.0d0 (log 2.0d0)))))))

(deftest test-expt-number-number
  (testing "expt delegates to cl:expt for numbers"
    (ok (= (ad:expt 2 3) 8))))

;;; Bug fix: log base is AD-transparent (M5)

(deftest test-log-derivative-wrt-base
  (testing "d/db[log_b(8)] at b=2 = -ln(8)/(b*ln(b)^2)"
    ;; log_b(x) = ln(x)/ln(b), chain rule:
    ;; d/db[ln(x)/ln(b)] = -ln(x)/(b*ln(b)^2)
    ;; at b=2, x=8: -ln(8)/(2*ln(2)^2)
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (b) (ad:log 8.0d0 b)) 2.0d0)
      (declare (ignore val))
      (let ((expected (/ (- (log 8.0d0)) (* 2.0d0 (expt (log 2.0d0) 2)))))
        (ok (approx= grad expected))))))

(deftest test-log-with-dual-base
  (testing "log_b(x) with dual base produces correct epsilon component"
    ;; d/db log_b(100) at b=10 = -ln(100)/(b*ln(b)^2) = -ln(100)/(10*ln(10)^2)
    (let ((result (ad:log 100.0d0 (ad:make-dual 10.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (log 100.0d0 10.0d0)))
      (ok (approx= (ad:dual-epsilon result)
                    (/ (- (log 100.0d0)) (* 10.0d0 (expt (log 10.0d0) 2))))))))

;;; Bug fix: expt with base=0 and dual power (M10)

(deftest test-expt-zero-base-dual-power
  (testing "0^(1+e) = (0.0, 0.0) - no NaN from 0*log(0)"
    (let ((result (ad:expt 0.0d0 (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) 0.0d0))
      (ok (approx= (ad:dual-epsilon result) 0.0d0)))))

(deftest test-expt-zero-base-via-derivative
  (testing "d/dy[0^y] at y=1 returns 0, not NaN"
    (let ((result (ad:derivative (lambda (y) (ad:expt 0.0d0 y)) 1.0d0)))
      (ok (approx= result 0.0d0)))))

;;; Bug fix regressions: M3, M4, M5

;;; M3: binary-expt (number, dual) with negative base

(deftest test-expt-negative-base-dual-power-no-type-error
  (testing "(-2)^(1+e) does not signal type-error; real part is finite (not NaN, not inf)"
    ;; For integer exponent 1: (-2)^1 = -2 (real); gradient zeroed (undefined for non-integer)
    (let* ((result (ad:expt -2.0d0 (ad:make-dual 1.0d0 1.0d0)))
           (r (ad:dual-real result)))
      (ok (typep result 'ad:dual))
      ;; NaN check: NaN is not equal to itself
      (ok (= r r))
      ;; Infinity check: finite value satisfies (<= (abs r) most-positive-double-float)
      (ok (<= (abs r) most-positive-double-float)))))

(deftest test-expt-negative-base-dual-power-integer-exponent
  (testing "(-2)^(3+0e): integer real part -> cl:expt returns exact real"
    ;; (-2)^3 = -8; gradient zeroed as sentinel
    (let ((result (ad:expt -2.0d0 (ad:make-dual 3.0d0 0.0d0))))
      (ok (approx= (ad:dual-real result) -8.0d0))
      (ok (approx= (ad:dual-epsilon result) 0.0d0)))))

;;; M4: binary-expt (dual, number) with dual real=0 and exponent < 1

(deftest test-expt-dual-zero-base-fractional-exponent-no-nan
  (testing "d/dx[x^0.5] at x=0 returns finite (not NaN)"
    (let ((result (ad:derivative (lambda (x) (ad:expt x 0.5d0)) 0.0d0)))
      (ok (typep result 'double-float))
      ;; NaN check: NaN /= NaN
      (ok (= result result)))))

(deftest test-expt-dual-zero-base-exponent-one
  (testing "(0+b*e)^1 = 0 + b*e: gradient passes through correctly"
    (let ((result (ad:expt (ad:make-dual 0.0d0 3.0d0) 1.0d0)))
      (ok (approx= (ad:dual-real result) 0.0d0))
      (ok (approx= (ad:dual-epsilon result) 3.0d0)))))

(deftest test-expt-dual-zero-base-negative-exponent-no-nan
  (testing "(0+e)^(-1): gradient is +inf mathematically; sentinel 0 returned, no NaN/error"
    (let* ((result (ad:expt (ad:make-dual 0.0d0 1.0d0) -1.0d0))
           (eps (ad:dual-epsilon result)))
      (ok (typep result 'ad:dual))
      ;; NaN check
      (ok (= eps eps)))))

;;; M5: unary-log (dual) with negative real part

(deftest test-log-negative-dual-no-type-error
  (testing "log(-1+e) does not signal type-error; gradient is zero sentinel"
    (let ((result (ad:log (ad:make-dual -1.0d0 1.0d0))))
      (ok (typep result 'ad:dual))
      (ok (approx= (ad:dual-epsilon result) 0.0d0)))))

(deftest test-log-zero-dual-no-type-error
  (testing "log(0+e) does not signal type-error; returns sentinel with zero gradient"
    (let ((result (ad:log (ad:make-dual 0.0d0 1.0d0))))
      (ok (typep result 'ad:dual))
      (ok (approx= (ad:dual-epsilon result) 0.0d0)))))

;;; Nonunit epsilon

(deftest test-sin-nonunit-epsilon
  (testing "sin with non-unit epsilon: sin(0.5+3e)"
    (let ((result (ad:sin (ad:make-dual 0.5d0 3.0d0))))
      (ok (approx= (ad:dual-real result) (sin 0.5d0)))
      (ok (approx= (ad:dual-epsilon result) (* (cos 0.5d0) 3.0d0))))))
