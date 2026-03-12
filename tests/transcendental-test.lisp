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

;;; Nonunit epsilon

(deftest test-sin-nonunit-epsilon
  (testing "sin with non-unit epsilon: sin(0.5+3e)"
    (let ((result (ad:sin (ad:make-dual 0.5d0 3.0d0))))
      (ok (approx= (ad:dual-real result) (sin 0.5d0)))
      (ok (approx= (ad:dual-epsilon result) (* (cos 0.5d0) 3.0d0))))))
