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
