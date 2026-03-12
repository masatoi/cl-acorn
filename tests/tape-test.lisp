(defpackage #:cl-acorn/tests/tape-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/tape-test)

(deftest test-make-tape-node
  (testing "tape-node stores value"
    (let ((n (make-instance 'ad:tape-node :value 3.0d0)))
      (ok (approx= (ad:node-value n) 3.0d0))
      (ok (approx= (ad:node-gradient n) 0)))))

(deftest test-tape-recording
  (testing "*tape* records nodes during computation"
    (let ((cl-acorn.ad::*tape* (list :start)))
      (let ((a (cl-acorn.ad::make-node 2.0d0 nil))
            (b (cl-acorn.ad::make-node 3.0d0 nil)))
        (declare (ignore a b))
        (ok (= (length cl-acorn.ad::*tape*) 3))))))

(deftest test-backward-simple-add
  (testing "backward propagates gradient through addition: z = a + b"
    (let ((cl-acorn.ad::*tape* (list :start)))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (z (cl-acorn.ad::make-node 5.0d0
                                        (list (cons a 1.0d0)
                                              (cons b 1.0d0)))))
        (setf cl-acorn.ad::*tape*
              (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient z) 1.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))
        (ok (approx= (ad:node-gradient b) 1.0d0))))))

(deftest test-backward-simple-mul
  (testing "backward propagates gradient through multiplication: z = a * b"
    (let ((cl-acorn.ad::*tape* (list :start)))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (z (cl-acorn.ad::make-node 6.0d0
                                        (list (cons a 3.0d0)
                                              (cons b 2.0d0)))))
        (setf cl-acorn.ad::*tape*
              (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient a) 3.0d0))
        (ok (approx= (ad:node-gradient b) 2.0d0))))))

(deftest test-backward-chain
  (testing "backward propagates through chain: z = (a * b) + b"
    (let ((cl-acorn.ad::*tape* (list :start)))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (ab (cl-acorn.ad::make-node 6.0d0
                                         (list (cons a 3.0d0)
                                               (cons b 2.0d0))))
             (z (cl-acorn.ad::make-node 9.0d0
                                        (list (cons ab 1.0d0)
                                              (cons b 1.0d0)))))
        (setf cl-acorn.ad::*tape*
              (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient a) 3.0d0))
        (ok (approx= (ad:node-gradient b) 3.0d0))))))

(deftest test-backward-resets-gradients
  (testing "backward sets root gradient to 1 and accumulates correctly"
    (let ((cl-acorn.ad::*tape* (list :start)))
      (let* ((a (cl-acorn.ad::make-node 5.0d0 nil))
             (z (cl-acorn.ad::make-node 5.0d0
                                        (list (cons a 1.0d0)))))
        (setf cl-acorn.ad::*tape*
              (butlast cl-acorn.ad::*tape*))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient z) 1.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))))))
