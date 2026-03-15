(defpackage #:cl-acorn/tests/vi-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/vi-test)

(deftest test-vi-standard-normal
  (testing "VI recovers standard normal posterior (mu~0, sigma~1)"
    (let ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))))
      (multiple-value-bind (mu sigma elbo)
          (infer:vi log-pdf '(0.0d0)
            :n-iterations 2000 :n-elbo-samples 10 :lr 0.05d0)
        (declare (ignore elbo))
        (ok (approx= (first mu) 0.0d0 0.3d0))
        (ok (approx= (first sigma) 1.0d0 0.3d0))))))

(deftest test-vi-shifted-normal
  (testing "VI recovers shifted normal posterior (mu~3, sigma~0.5)"
    ;; N(3, 0.5): log-pdf = -0.5 * ((x-3)/0.5)^2 - log(0.5)
    (let ((log-pdf (lambda (p)
                     (dist:normal-log-pdf (first p) :mu 3.0d0 :sigma 0.5d0))))
      (multiple-value-bind (mu sigma elbo)
          (infer:vi log-pdf '(0.0d0)
            :n-iterations 2000 :n-elbo-samples 10 :lr 0.05d0)
        (declare (ignore elbo))
        (ok (approx= (first mu) 3.0d0 0.3d0))
        (ok (approx= (first sigma) 0.5d0 0.2d0))))))

(deftest test-vi-multivariate
  (testing "VI recovers 2D independent normals"
    ;; N(0,1) x N(2,1)
    (let ((log-pdf (lambda (p)
                     (ad:+ (ad:* -0.5d0 (ad:* (first p) (first p)))
                           (let ((shifted (ad:- (second p) 2.0d0)))
                             (ad:* -0.5d0 (ad:* shifted shifted)))))))
      (multiple-value-bind (mu sigma elbo)
          (infer:vi log-pdf '(0.0d0 0.0d0)
            :n-iterations 2000 :n-elbo-samples 10 :lr 0.05d0)
        (declare (ignore elbo))
        (ok (approx= (first mu) 0.0d0 0.3d0))
        (ok (approx= (second mu) 2.0d0 0.3d0))
        (ok (approx= (first sigma) 1.0d0 0.3d0))
        (ok (approx= (second sigma) 1.0d0 0.3d0))))))

(deftest test-vi-elbo-increasing
  (testing "ELBO improves from early to later iterations"
    ;; Use a shifted normal for a stronger initial signal
    (let ((log-pdf (lambda (p)
                     (dist:normal-log-pdf (first p) :mu 5.0d0 :sigma 1.0d0))))
      (multiple-value-bind (mu sigma elbo)
          (infer:vi log-pdf '(0.0d0)
            :n-iterations 1000 :n-elbo-samples 20 :lr 0.05d0)
        (declare (ignore mu sigma))
        ;; Compare first 50 vs iterations 400-500 for robust signal
        (let* ((early (/ (reduce #'+ (subseq elbo 0 50)) 50.0d0))
               (later (/ (reduce #'+ (subseq elbo 400 500)) 100.0d0)))
          ;; Later ELBO should be higher than early ELBO
          (ok (> later early)))))))
