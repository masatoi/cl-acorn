(defpackage #:cl-acorn/tests/reverse-arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/reverse-arithmetic-test)

;;; Helper: compute reverse-mode gradient for f(a,b) and return (da db)
(defun reverse-grad-2 (fn a-val b-val)
  "Compute reverse-mode gradients of binary function FN at (A-VAL, B-VAL).
Returns (values f-val da db)."
  (let ((cl-acorn.ad::*tape* (list t)))
    (let* ((a (cl-acorn.ad::make-node (coerce a-val 'double-float) nil))
           (b (cl-acorn.ad::make-node (coerce b-val 'double-float) nil))
           (result (funcall fn a b)))
      (setf cl-acorn.ad::*tape* (butlast cl-acorn.ad::*tape*))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient a)
              (ad:node-gradient b)))))

;;; Helper: compute reverse-mode gradient for f(x) and return dx
(defun reverse-grad-1 (fn x-val)
  "Compute reverse-mode gradient of unary function FN at X-VAL.
Returns (values f-val dx)."
  (let ((cl-acorn.ad::*tape* (list t)))
    (let* ((x (cl-acorn.ad::make-node (coerce x-val 'double-float) nil))
           (result (funcall fn x)))
      (setf cl-acorn.ad::*tape* (butlast cl-acorn.ad::*tape*))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient x)))))

;;; --- Addition ---

(deftest test-reverse-add-tape-tape
  (testing "d/da(a+b)=1, d/db(a+b)=1"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:+ 2.0d0 3.0d0)
      (ok (approx= val 5.0d0))
      (ok (approx= da 1.0d0))
      (ok (approx= db 1.0d0)))))

(deftest test-reverse-add-tape-number
  (testing "d/da(a+5) = 1"
    (multiple-value-bind (val da)
        (reverse-grad-1 (lambda (a) (ad:+ a 5.0d0)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= da 1.0d0)))))

(deftest test-reverse-add-number-tape
  (testing "d/db(5+b) = 1"
    (multiple-value-bind (val db)
        (reverse-grad-1 (lambda (b) (ad:+ 5.0d0 b)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= db 1.0d0)))))

;;; --- Subtraction ---

(deftest test-reverse-sub-tape-tape
  (testing "d/da(a-b)=1, d/db(a-b)=-1"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:- 5.0d0 3.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= da 1.0d0))
      (ok (approx= db -1.0d0)))))

(deftest test-reverse-negate
  (testing "d/dx(-x) = -1"
    (multiple-value-bind (val dx)
        (reverse-grad-1 #'ad:- 3.0d0)
      (ok (approx= val -3.0d0))
      (ok (approx= dx -1.0d0)))))

;;; --- Multiplication ---

(deftest test-reverse-mul-tape-tape
  (testing "d/da(a*b)=b, d/db(a*b)=a"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:* 2.0d0 3.0d0)
      (ok (approx= val 6.0d0))
      (ok (approx= da 3.0d0))
      (ok (approx= db 2.0d0)))))

(deftest test-reverse-mul-tape-number
  (testing "d/da(a*5) = 5"
    (multiple-value-bind (val da)
        (reverse-grad-1 (lambda (a) (ad:* a 5.0d0)) 3.0d0)
      (ok (approx= val 15.0d0))
      (ok (approx= da 5.0d0)))))

(deftest test-reverse-mul-number-tape
  (testing "d/db(5*b) = 5"
    (multiple-value-bind (val db)
        (reverse-grad-1 (lambda (b) (ad:* 5.0d0 b)) 3.0d0)
      (ok (approx= val 15.0d0))
      (ok (approx= db 5.0d0)))))

;;; --- Division ---

(deftest test-reverse-div-tape-tape
  (testing "d/da(a/b)=1/b, d/db(a/b)=-a/b^2"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:/ 6.0d0 3.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= da (/ 1.0d0 3.0d0)))
      (ok (approx= db (/ -6.0d0 9.0d0))))))

(deftest test-reverse-reciprocal
  (testing "d/dx(1/x) = -1/x^2"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:/ x)) 2.0d0)
      (ok (approx= val 0.5d0))
      (ok (approx= dx -0.25d0)))))

;;; --- N-ary wrappers ---

(deftest test-reverse-add-nary
  (testing "gradient through 3-arg addition"
    (let ((cl-acorn.ad::*tape* (list t)))
      (let* ((a (cl-acorn.ad::make-node 1.0d0 nil))
             (b (cl-acorn.ad::make-node 2.0d0 nil))
             (c (cl-acorn.ad::make-node 3.0d0 nil))
             (result (ad:+ a b c)))
        (setf cl-acorn.ad::*tape* (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward result)
        (ok (approx= (ad:node-value result) 6.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))
        (ok (approx= (ad:node-gradient b) 1.0d0))
        (ok (approx= (ad:node-gradient c) 1.0d0))))))

(deftest test-reverse-mul-nary
  (testing "gradient through 3-arg multiplication"
    (let ((cl-acorn.ad::*tape* (list t)))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (c (cl-acorn.ad::make-node 4.0d0 nil))
             (result (ad:* a b c)))
        (setf cl-acorn.ad::*tape* (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward result)
        (ok (approx= (ad:node-value result) 24.0d0))
        (ok (approx= (ad:node-gradient a) 12.0d0))
        (ok (approx= (ad:node-gradient b) 8.0d0))
        (ok (approx= (ad:node-gradient c) 6.0d0))))))

;;; --- Cross-validation with forward-mode ---

(deftest test-reverse-matches-forward-polynomial
  (testing "reverse gradient of x^3-2x+5 matches forward derivative at x=3"
    (let ((fn (lambda (x) (ad:+ (ad:* x x x) (ad:+ (ad:* -2 x) 5)))))
      (multiple-value-bind (fwd-val fwd-grad)
          (ad:derivative fn 3.0d0)
        (multiple-value-bind (rev-val rev-grad)
            (reverse-grad-1 fn 3.0d0)
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad rev-grad)))))))
