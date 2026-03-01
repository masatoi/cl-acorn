(defpackage #:cl-acorn/tests/arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

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

(deftest test-sub-number-number
  (testing "5 - 3 = 2 (falls through to cl:-)"
    (ok (= (ad:- 5 3) 2))))
