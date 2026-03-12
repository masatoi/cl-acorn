(defpackage #:cl-acorn/tests/distributions-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/distributions-test)

;;; --- log-gammaln tests ---

(deftest test-log-gammaln-integers
  (testing "log-gammaln at integer values matches log((n-1)!)"
    (ok (approx= (dist:log-gammaln 1.0d0) 0.0d0 1d-10))          ; Γ(1) = 0! = 1
    (ok (approx= (dist:log-gammaln 2.0d0) 0.0d0 1d-10))          ; Γ(2) = 1! = 1
    (ok (approx= (dist:log-gammaln 3.0d0) (log 2.0d0) 1d-10))    ; Γ(3) = 2! = 2
    (ok (approx= (dist:log-gammaln 5.0d0) (log 24.0d0) 1d-10))   ; Γ(5) = 4! = 24
    (ok (approx= (dist:log-gammaln 7.0d0) (log 720.0d0) 1d-10)))) ; Γ(7) = 6! = 720

(deftest test-log-gammaln-half
  (testing "log-gammaln at 0.5 equals log(sqrt(pi))"
    (ok (approx= (dist:log-gammaln 0.5d0)
                 (* 0.5d0 (log pi))
                 1d-10))))

;;; --- normal distribution tests ---

(deftest test-normal-log-pdf-standard
  (testing "standard normal log-pdf at known points"
    ;; log N(0|0,1) = -0.5*log(2pi) ~ -0.9189
    (ok (approx= (dist:normal-log-pdf 0.0d0)
                 -0.9189385332046727d0 1d-10))
    ;; log N(1|0,1) = -0.5*log(2pi) - 0.5 ~ -1.4189
    (ok (approx= (dist:normal-log-pdf 1.0d0)
                 -1.4189385332046727d0 1d-10))))

(deftest test-normal-log-pdf-nonstandard
  (testing "normal log-pdf with explicit mu and sigma"
    ;; log N(3|2,0.5): z=(3-2)/0.5=2, -0.5*log(2pi) - log(0.5) - 0.5*4
    (let ((expected (- (- (* -0.5d0 (log (* 2.0d0 pi)))
                          (log 0.5d0))
                       2.0d0)))
      (ok (approx= (dist:normal-log-pdf 3.0d0 :mu 2.0d0 :sigma 0.5d0)
                   expected 1d-10)))))

(deftest test-normal-log-pdf-ad-forward
  (testing "normal log-pdf differentiable via forward-mode"
    ;; d/dx log N(x|0,1) at x=1 = -(x-mu)/sigma^2 = -1.0
    (multiple-value-bind (val deriv)
        (ad:derivative (lambda (x) (dist:normal-log-pdf x)) 1.0d0)
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= deriv -1.0d0 1d-10)))))

(deftest test-normal-log-pdf-ad-reverse
  (testing "normal log-pdf differentiable via reverse-mode (gradient w.r.t. mu)"
    ;; d/dmu log N(1|mu,1) at mu=0 = (x-mu)/sigma^2 = 1.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:normal-log-pdf 1.0d0 :mu (first p) :sigma 1.0d0))
                     '(0.0d0))
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= (first grad) 1.0d0 1d-10)))))

(deftest test-normal-sample-range
  (testing "normal samples have reasonable mean"
    (let* ((n 10000)
           (samples (loop repeat n collect (dist:normal-sample :mu 5.0d0 :sigma 0.1d0)))
           (mean (/ (reduce #'cl:+ samples) n)))
      (ok (approx= mean 5.0d0 0.05d0)))))
