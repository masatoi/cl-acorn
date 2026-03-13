(defpackage #:cl-acorn/tests/nuts-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/nuts-test)

(deftest test-nuts-standard-normal
  (testing "NUTS samples from standard normal have correct mean and variance"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 1000)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:nuts log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0)
        (declare (ignore accept-rate))
        ;; Check mean ~ 0
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.2d0)))
        ;; Check variance ~ 1 (NUTS can have autocorrelation, so widen tolerance)
        (let* ((mean (/ (reduce #'+ samples :key #'first) n-samples))
               (var (/ (reduce #'+ samples
                               :key (lambda (s) (expt (- (first s) mean) 2)))
                       n-samples)))
          (ok (approx= var 1.0d0 0.5d0)))))))

(deftest test-nuts-bivariate-normal
  (testing "NUTS samples from 2D independent normals"
    ;; N(0,1) x N(3,1)
    (let* ((log-pdf (lambda (p)
                      (ad:+ (ad:* -0.5d0 (ad:* (first p) (first p)))
                            (let ((shifted (ad:- (second p) 3.0d0)))
                              (ad:* -0.5d0 (ad:* shifted shifted))))))
           (n-samples 1000)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:nuts log-pdf '(0.0d0 0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0)
        (declare (ignore accept-rate))
        ;; Check dim 1 mean ~ 0
        (let ((mean1 (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean1 0.0d0 0.2d0)))
        ;; Check dim 2 mean ~ 3
        (let ((mean2 (/ (reduce #'+ samples :key #'second) n-samples)))
          (ok (approx= mean2 3.0d0 0.2d0)))))))

(deftest test-nuts-adapt-step-size
  (testing "NUTS adaptation rescues a bad initial step-size"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 500)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:nuts log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 1.0d0 :adapt-step-size t)
        (declare (ignore accept-rate))
        ;; Mean should be approximately correct despite bad initial step-size
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.3d0)))))))

(deftest test-nuts-max-tree-depth
  (testing "NUTS respects max-tree-depth limit"
    ;; With depth=1, max 2 leapfrog steps per iteration
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 200)
           (n-warmup 100))
      ;; Should still run and produce samples (just less efficient)
      (multiple-value-bind (samples accept-rate)
          (infer:nuts log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0 :max-tree-depth 2
            :adapt-step-size nil)
        (declare (ignore accept-rate))
        (ok (= (length samples) n-samples))))))
