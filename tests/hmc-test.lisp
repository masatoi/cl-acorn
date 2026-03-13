(defpackage #:cl-acorn/tests/hmc-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/hmc-test)

(deftest test-hmc-leapfrog-energy-conservation
  (testing "leapfrog approximately conserves Hamiltonian energy"
    ;; Use standard normal: log-pdf(x) = -0.5*x^2 (ignore constant)
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (q '(1.0d0))
           (p-mom '(0.5d0))
           ;; Hamiltonian = -log-pdf(q) + 0.5*p^2 = 0.5*q^2 + 0.5*p^2
           (h-initial (+ (* 0.5d0 1.0d0 1.0d0) (* 0.5d0 0.5d0 0.5d0))))
      (multiple-value-bind (q-new p-new)
          (cl-acorn.inference::leapfrog log-pdf q p-mom 0.1d0 20)
        (let ((h-final (+ (* 0.5d0 (first q-new) (first q-new))
                          (* 0.5d0 (first p-new) (first p-new)))))
          ;; Energy should be approximately conserved
          (ok (approx= h-initial h-final 0.01d0)))))))

(deftest test-hmc-standard-normal
  (testing "HMC samples from standard normal have correct mean and variance"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 2000)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0 :n-leapfrog 20)
        ;; Accept rate should be reasonable (> 50%)
        (ok (> accept-rate 0.5d0))
        ;; Check mean ≈ 0
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.15d0)))
        ;; Check variance ≈ 1
        (let* ((mean (/ (reduce #'+ samples :key #'first) n-samples))
               (var (/ (reduce #'+ samples
                               :key (lambda (s) (expt (- (first s) mean) 2)))
                       n-samples)))
          (ok (approx= var 1.0d0 0.2d0)))))))

(deftest test-hmc-with-distributions
  (testing "HMC works with dist:normal-log-pdf for a shifted normal"
    ;; Sample from N(3, 1) using dist:normal-log-pdf
    (let ((log-pdf (lambda (p)
                     (dist:normal-log-pdf (first p) :mu 3.0d0 :sigma 1.0d0))))
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples 2000 :n-warmup 500
            :step-size 0.1d0 :n-leapfrog 20)
        (declare (ignore accept-rate))
        (let ((mean (/ (reduce #'+ samples :key #'first) 2000)))
          (ok (approx= mean 3.0d0 0.2d0)))))))

(deftest test-hmc-leapfrog-step
  (testing "leapfrog-step performs a single integration step"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (q '(1.0d0))
           (p-mom '(0.5d0)))
      (multiple-value-bind (q-new p-new log-pdf-new grad-new)
          (cl-acorn.inference::leapfrog-step log-pdf q p-mom 0.1d0)
        (declare (ignore log-pdf-new grad-new))
        ;; After one step, position should have moved
        (ok (not (approx= (first q-new) 1.0d0 1d-10)))
        ;; Momentum should have changed
        (ok (not (approx= (first p-new) 0.5d0 1d-10)))))))

(deftest test-hmc-adapt-step-size
  (testing "HMC with step-size adaptation rescues a bad initial step-size"
    ;; Use a deliberately bad step-size (1.0) which would give terrible
    ;; accept rate without adaptation
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 1000)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 1.0d0 :n-leapfrog 10
            :adapt-step-size t)
        ;; Adaptation should produce reasonable accept rate (> 40%)
        (ok (> accept-rate 0.4d0))
        ;; Mean should still be approximately correct
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.2d0)))))))

(deftest test-hmc-adapt-backward-compat
  (testing "HMC without adapt-step-size flag behaves as before"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 500)
           (n-warmup 200))
      ;; Good step-size, no adaptation - should work fine
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0 :n-leapfrog 20)
        (ok (> accept-rate 0.5d0))
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.2d0)))))))
