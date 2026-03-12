(defpackage #:cl-acorn/tests/arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/arithmetic-test)

;;; Addition tests

(deftest test-add-dual-dual
  (testing "(1+2e) + (3+4e) = (4+6e)"
    (let ((result (ad:+ (ad:make-dual 1 2) (ad:make-dual 3 4))))
      (ok (typep result 'ad:dual))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-add-dual-number
  (testing "(1+2e) + 3 = (4+2e)"
    (let ((result (ad:+ (ad:make-dual 1 2) 3)))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-add-number-dual
  (testing "3 + (1+2e) = (4+2e)"
    (let ((result (ad:+ 3 (ad:make-dual 1 2))))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-add-number-number
  (testing "3 + 4 = 7 (falls through to cl:+)"
    (ok (= (ad:+ 3 4) 7))))

(deftest test-add-n-ary
  (testing "n-ary addition"
    (let ((result (ad:+ (ad:make-dual 1 1) (ad:make-dual 2 2) (ad:make-dual 3 3))))
      (ok (approx= (ad:dual-real result) 6.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-add-zero-args
  (testing "(ad:+) returns 0"
    (ok (= (ad:+) 0))))

(deftest test-add-one-arg
  (testing "(ad:+ x) returns x"
    (let ((d (ad:make-dual 5 3)))
      (ok (eq (ad:+ d) d)))))

;;; Subtraction tests

(deftest test-sub-dual-dual
  (testing "(5+3e) - (2+1e) = (3+2e)"
    (let ((result (ad:- (ad:make-dual 5 3) (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-sub-dual-number
  (testing "(5+3e) - 2 = (3+3e)"
    (let ((result (ad:- (ad:make-dual 5 3) 2)))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 3.0d0)))))

(deftest test-sub-number-dual
  (testing "5 - (2+3e) = (3 + -3e)"
    (let ((result (ad:- 5 (ad:make-dual 2 3))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) -3.0d0)))))

(deftest test-sub-negation
  (testing "(ad:- x) negates dual"
    (let ((result (ad:- (ad:make-dual 3 5))))
      (ok (approx= (ad:dual-real result) -3.0d0))
      (ok (approx= (ad:dual-epsilon result) -5.0d0)))))

(deftest test-negate-number
  (testing "(ad:- 5) negates plain number"
    (ok (= (ad:- 5) -5))))

(deftest test-negate-derivative
  (testing "d/dx(-x) = -1 at x=3"
    (multiple-value-bind (val deriv)
        (ad:derivative (lambda (x) (ad:- x)) 3.0d0)
      (ok (approx= val -3.0d0))
      (ok (approx= deriv -1.0d0)))))

(deftest test-sub-number-number
  (testing "5 - 3 = 2 (falls through to cl:-)"
    (ok (= (ad:- 5 3) 2))))

;;; Multiplication tests

(deftest test-mul-dual-dual
  (testing "(2+3e) * (4+5e) = (8 + 22e)"
    ;; real: 2*4=8, eps: 2*5 + 3*4 = 10+12 = 22
    (let ((result (ad:* (ad:make-dual 2 3) (ad:make-dual 4 5))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 22.0d0)))))

(deftest test-mul-dual-number
  (testing "(2+3e) * 4 = (8+12e)"
    (let ((result (ad:* (ad:make-dual 2 3) 4)))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-mul-number-dual
  (testing "4 * (2+3e) = (8+12e)"
    (let ((result (ad:* 4 (ad:make-dual 2 3))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-mul-number-number
  (testing "3 * 4 = 12"
    (ok (= (ad:* 3 4) 12))))

(deftest test-mul-n-ary
  (testing "n-ary multiplication"
    ;; (1+1e) * (2+0e) * (3+0e) = first: (2+2e), then: (6+6e)
    (let ((result (ad:* (ad:make-dual 1 1) (ad:make-dual 2 0) (ad:make-dual 3 0))))
      (ok (approx= (ad:dual-real result) 6.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-mul-zero-args
  (testing "(ad:*) returns 1"
    (ok (= (ad:*) 1))))

;;; Division tests

(deftest test-div-dual-dual
  (testing "(6+4e) / (3+1e) = (2 + 2/3e)"
    ;; real: 6/3=2, eps: (4*3 - 6*1)/9 = 6/9 = 2/3
    (let ((result (ad:/ (ad:make-dual 6 4) (ad:make-dual 3 1))))
      (ok (approx= (ad:dual-real result) 2.0d0))
      (ok (approx= (ad:dual-epsilon result) (/ 2.0d0 3.0d0))))))

(deftest test-div-dual-number
  (testing "(6+4e) / 2 = (3+2e)"
    (let ((result (ad:/ (ad:make-dual 6 4) 2)))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-div-number-dual
  (testing "1 / (2+1e) = (0.5 + -0.25e)"
    ;; real: 1/2=0.5, eps: -(1*1)/4 = -0.25
    (let ((result (ad:/ 1 (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 0.5d0))
      (ok (approx= (ad:dual-epsilon result) -0.25d0)))))

(deftest test-div-reciprocal
  (testing "(ad:/ x) computes reciprocal"
    ;; 1/(2+1e) = 0.5 - 0.25e
    (let ((result (ad:/ (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 0.5d0))
      (ok (approx= (ad:dual-epsilon result) -0.25d0)))))

(deftest test-reciprocal-number
  (testing "(ad:/ 5) computes reciprocal of plain number"
    (ok (= (ad:/ 5) 1/5))))

(deftest test-reciprocal-derivative
  (testing "d/dx(1/x) = -1/x^2 at x=2"
    ;; f(x) = 1/x, f'(x) = -1/x^2
    ;; At x=2: f(2) = 0.5, f'(2) = -0.25
    (multiple-value-bind (val deriv)
        (ad:derivative (lambda (x) (ad:/ x)) 2.0d0)
      (ok (approx= val 0.5d0))
      (ok (approx= deriv -0.25d0)))))

(deftest test-div-number-number
  (testing "6 / 3 = 2"
    (ok (= (ad:/ 6 3) 2))))
