(defpackage #:cl-acorn/tests/dual-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/dual-test)

(deftest test-make-dual-defaults
  (testing "make-dual with default epsilon"
    (let ((d (ad:make-dual 3.0d0)))
      (ok (typep d 'ad:dual))
      (ok (approx= (ad:dual-real d) 3.0d0))
      (ok (approx= (ad:dual-epsilon d) 0.0d0)))))

(deftest test-make-dual-both-args
  (testing "make-dual with real and epsilon"
    (let ((d (ad:make-dual 2.0d0 5.0d0)))
      (ok (approx= (ad:dual-real d) 2.0d0))
      (ok (approx= (ad:dual-epsilon d) 5.0d0)))))

(deftest test-make-dual-coercion
  (testing "make-dual coerces integers to double-float"
    (let ((d (ad:make-dual 3 7)))
      (ok (typep (ad:dual-real d) 'double-float))
      (ok (typep (ad:dual-epsilon d) 'double-float))
      (ok (approx= (ad:dual-real d) 3.0d0))
      (ok (approx= (ad:dual-epsilon d) 7.0d0)))))

(deftest test-make-dual-rational-coercion
  (testing "make-dual coerces rationals to double-float"
    (let ((d (ad:make-dual 1/3 1)))
      (ok (typep (ad:dual-real d) 'double-float))
      (ok (approx= (ad:dual-real d) (coerce 1/3 'double-float))))))

(deftest test-dual-print
  (testing "print-object produces readable representation"
    (let ((output (format nil "~A" (ad:make-dual 1.0d0 2.0d0))))
      (ok (search "DUAL" output))
      (ok (search "1.0d0" output))
      (ok (search "2.0d0" output)))))
