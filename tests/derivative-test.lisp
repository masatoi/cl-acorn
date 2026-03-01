(defpackage #:cl-acorn/tests/derivative-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/derivative-test)

(deftest test-derivative-identity
  (testing "d/dx[x] = 1"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) x) 5.0d0)
      (ok (approx= val 5.0d0))
      (ok (approx= grad 1.0d0)))))

(deftest test-derivative-constant
  (testing "d/dx[42] = 0"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (declare (ignore x)) 42.0d0) 5.0d0)
      (ok (approx= val 42.0d0))
      (ok (approx= grad 0.0d0)))))

(deftest test-derivative-square
  (testing "d/dx[x^2] = 2x at x=3"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:* x x)) 3.0d0)
      (ok (approx= val 9.0d0))
      (ok (approx= grad 6.0d0)))))

(deftest test-derivative-polynomial
  (testing "d/dx[3x^2 + 2x + 1] at x=2"
    ;; f(2) = 12+4+1 = 17, f'(2) = 6*2+2 = 14
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:+ (ad:* 3 (ad:* x x)) (ad:+ (ad:* 2 x) 1)))
         2.0d0)
      (ok (approx= val 17.0d0))
      (ok (approx= grad 14.0d0)))))

(deftest test-derivative-sin
  (testing "d/dx[sin(x)] = cos(x) at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:sin x)) 1.0d0)
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= grad (cos 1.0d0))))))

(deftest test-derivative-composite-1
  (testing "d/dx[sin(x^2) + exp(2x)/x] at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x)
           (ad:+ (ad:sin (ad:* x x))
                 (ad:/ (ad:exp (ad:* 2 x)) x)))
         1.0d0)
      ;; f(1) = sin(1) + exp(2)
      (let ((expected-val (+ (sin 1.0d0) (exp 2.0d0))))
        (ok (approx= val expected-val)))
      ;; f'(1) = 2*cos(1) + exp(2)*(2*1-1)/1^2
      (let ((expected-grad (+ (* 2.0d0 (cos 1.0d0))
                              (* (exp 2.0d0)
                                 (/ (- (* 2.0d0 1.0d0) 1.0d0)
                                    (* 1.0d0 1.0d0))))))
        (ok (approx= grad expected-grad))))))

(deftest test-derivative-composite-2
  (testing "d/dx[log(x) * sqrt(x)] at x=2"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:* (ad:log x) (ad:sqrt x)))
         2.0d0)
      ;; f(2) = ln(2) * sqrt(2)
      (ok (approx= val (* (log 2.0d0) (sqrt 2.0d0))))
      ;; f'(x) = (2 + ln(x)) / (2*sqrt(x))
      (let ((expected-grad (/ (+ 2.0d0 (log 2.0d0))
                              (* 2.0d0 (sqrt 2.0d0)))))
        (ok (approx= grad expected-grad))))))

(deftest test-derivative-exp-chain
  (testing "d/dx[exp(sin(x))] at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:exp (ad:sin x)))
         1.0d0)
      ;; f(1) = exp(sin(1))
      (ok (approx= val (exp (sin 1.0d0))))
      ;; f'(1) = exp(sin(1)) * cos(1)
      (ok (approx= grad (* (exp (sin 1.0d0)) (cos 1.0d0)))))))

(deftest test-derivative-integer-input
  (testing "derivative accepts integer input"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:* x x)) 3)
      (ok (approx= val 9.0d0))
      (ok (approx= grad 6.0d0)))))
