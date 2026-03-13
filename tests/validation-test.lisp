(defpackage #:cl-acorn/tests/validation-test
  (:use #:cl #:rove))

(in-package #:cl-acorn/tests/validation-test)

;;; --- Domain guards for log-pdf functions ---

(deftest test-gamma-log-pdf-domain-guard
  (testing "gamma-log-pdf returns sentinel for x <= 0"
    (ok (= (dist:gamma-log-pdf -1.0d0 :shape 2.0d0 :rate 1.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:gamma-log-pdf 0.0d0 :shape 2.0d0 :rate 1.0d0)
            dist:+log-pdf-sentinel+)))
  (testing "gamma-log-pdf returns sentinel for rate <= 0"
    (ok (= (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 0.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate -1.0d0)
            dist:+log-pdf-sentinel+)))
  (testing "gamma-log-pdf returns finite value for valid inputs"
    (ok (< dist:+log-pdf-sentinel+
           (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 1.0d0)))))

(deftest test-beta-log-pdf-domain-guard
  (testing "beta-log-pdf returns sentinel for x outside (0, 1)"
    (ok (= (dist:beta-log-pdf -0.5d0 :alpha 2.0d0 :beta 2.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:beta-log-pdf 0.0d0 :alpha 2.0d0 :beta 2.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:beta-log-pdf 1.0d0 :alpha 2.0d0 :beta 2.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:beta-log-pdf 1.5d0 :alpha 2.0d0 :beta 2.0d0)
            dist:+log-pdf-sentinel+)))
  (testing "beta-log-pdf returns finite value for valid x"
    (ok (< dist:+log-pdf-sentinel+
           (dist:beta-log-pdf 0.5d0 :alpha 2.0d0 :beta 2.0d0)))))

(deftest test-normal-log-pdf-domain-guard
  (testing "normal-log-pdf returns sentinel for sigma <= 0"
    (ok (= (dist:normal-log-pdf 1.0d0 :sigma 0.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:normal-log-pdf 1.0d0 :sigma -1.0d0)
            dist:+log-pdf-sentinel+)))
  (testing "normal-log-pdf returns finite value for valid sigma"
    (ok (< dist:+log-pdf-sentinel+
           (dist:normal-log-pdf 1.0d0 :sigma 1.0d0)))))

(deftest test-bernoulli-log-pdf-domain-guard
  (testing "bernoulli-log-pdf returns sentinel for prob outside [0, 1]"
    (ok (= (dist:bernoulli-log-pdf 1 :prob -0.5d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:bernoulli-log-pdf 1 :prob 1.5d0)
            dist:+log-pdf-sentinel+)))
  (testing "bernoulli-log-pdf returns sentinel for x not in {0, 1}"
    (ok (= (dist:bernoulli-log-pdf -1.0d0 :prob 0.5d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:bernoulli-log-pdf 0.5d0 :prob 0.5d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:bernoulli-log-pdf 5.0d0 :prob 0.3d0)
            dist:+log-pdf-sentinel+)))
  (testing "bernoulli-log-pdf handles prob=0 boundary"
    (ok (= (dist:bernoulli-log-pdf 0.0d0 :prob 0.0d0) 0.0d0))
    (ok (= (dist:bernoulli-log-pdf 1.0d0 :prob 0.0d0) dist:+log-pdf-sentinel+)))
  (testing "bernoulli-log-pdf handles prob=1 boundary"
    (ok (= (dist:bernoulli-log-pdf 1.0d0 :prob 1.0d0) 0.0d0))
    (ok (= (dist:bernoulli-log-pdf 0.0d0 :prob 1.0d0) dist:+log-pdf-sentinel+)))
  (testing "bernoulli-log-pdf returns finite value for valid inputs"
    (ok (< dist:+log-pdf-sentinel+
           (dist:bernoulli-log-pdf 1 :prob 0.5d0)))
    (ok (< dist:+log-pdf-sentinel+
           (dist:bernoulli-log-pdf 0 :prob 0.5d0)))))

(deftest test-poisson-log-pdf-domain-guard
  (testing "poisson-log-pdf returns sentinel for rate <= 0"
    (ok (= (dist:poisson-log-pdf 3 :rate 0.0d0)
            dist:+log-pdf-sentinel+))
    (ok (= (dist:poisson-log-pdf 3 :rate -1.0d0)
            dist:+log-pdf-sentinel+)))
  (testing "poisson-log-pdf signals error for non-integer or negative k"
    (ok (signals (dist:poisson-log-pdf -1 :rate 1.0d0)))
    (ok (signals (dist:poisson-log-pdf 0.5d0 :rate 1.0d0))))
  (testing "poisson-log-pdf accepts integer-valued floats"
    (ok (< dist:+log-pdf-sentinel+
           (dist:poisson-log-pdf 3.0d0 :rate 2.0d0))))
  (testing "poisson-log-pdf returns finite value for valid inputs"
    (ok (< dist:+log-pdf-sentinel+
           (dist:poisson-log-pdf 3 :rate 2.0d0)))))

(deftest test-uniform-log-pdf-validates-bounds
  (testing "uniform-log-pdf signals error when low >= high"
    (ok (signals (dist:uniform-log-pdf 0.5d0 :low 1.0d0 :high 1.0d0)))
    (ok (signals (dist:uniform-log-pdf 0.5d0 :low 2.0d0 :high 1.0d0)))))

(deftest test-sentinel-safe-to-sum
  (testing "sentinel can be summed without overflow"
    (let ((sum (+ dist:+log-pdf-sentinel+ dist:+log-pdf-sentinel+
                  dist:+log-pdf-sentinel+ dist:+log-pdf-sentinel+)))
      (ok (< sum 0.0d0))
      (ok (not (= sum most-negative-double-float))))))

;;; --- Parameter validation for log-pdf functions ---

(deftest test-gamma-log-pdf-validates-shape
  (testing "gamma-log-pdf signals error for non-positive shape"
    (ok (signals (dist:gamma-log-pdf 1.0d0 :shape 0.0d0 :rate 1.0d0)))
    (ok (signals (dist:gamma-log-pdf 1.0d0 :shape -1.0d0 :rate 1.0d0)))))

(deftest test-beta-log-pdf-validates-params
  (testing "beta-log-pdf signals error for non-positive alpha"
    (ok (signals (dist:beta-log-pdf 0.5d0 :alpha 0.0d0 :beta 1.0d0))))
  (testing "beta-log-pdf signals error for non-positive beta"
    (ok (signals (dist:beta-log-pdf 0.5d0 :alpha 1.0d0 :beta 0.0d0)))))

;;; --- Sample function validation ---

(deftest test-normal-sample-validates-sigma
  (testing "normal-sample signals error for non-positive sigma"
    (ok (signals (dist:normal-sample :sigma 0)))
    (ok (signals (dist:normal-sample :sigma -1)))))

(deftest test-gamma-sample-validates-params
  (testing "gamma-sample signals error for non-positive shape"
    (ok (signals (dist:gamma-sample :shape 0 :rate 1.0d0)))
    (ok (signals (dist:gamma-sample :shape -1 :rate 1.0d0))))
  (testing "gamma-sample signals error for non-positive rate"
    (ok (signals (dist:gamma-sample :shape 1.0d0 :rate 0)))
    (ok (signals (dist:gamma-sample :shape 1.0d0 :rate -1)))))

(deftest test-bernoulli-sample-validates-prob
  (testing "bernoulli-sample signals error for prob outside [0, 1]"
    (ok (signals (dist:bernoulli-sample :prob -0.1)))
    (ok (signals (dist:bernoulli-sample :prob 1.1)))))

(deftest test-poisson-sample-validates-rate
  (testing "poisson-sample signals error for non-positive rate"
    (ok (signals (dist:poisson-sample :rate 0)))
    (ok (signals (dist:poisson-sample :rate -1))))
  (testing "poisson-sample signals error for rate > 700"
    (ok (signals (dist:poisson-sample :rate 701)))
    (ok (signals (dist:poisson-sample :rate 1000)))))

(deftest test-uniform-sample-validates-bounds
  (testing "uniform-sample signals error when low >= high"
    (ok (signals (dist:uniform-sample :low 1.0d0 :high 1.0d0)))
    (ok (signals (dist:uniform-sample :low 2.0d0 :high 1.0d0)))))

(deftest test-beta-sample-validates-params
  (testing "beta-sample signals error for non-positive alpha"
    (ok (signals (dist:beta-sample :alpha 0.0d0 :beta 1.0d0)))
    (ok (signals (dist:beta-sample :alpha -1.0d0 :beta 1.0d0))))
  (testing "beta-sample signals error for non-positive beta"
    (ok (signals (dist:beta-sample :alpha 1.0d0 :beta 0.0d0)))
    (ok (signals (dist:beta-sample :alpha 1.0d0 :beta -1.0d0)))))

;;; --- log-gammaln validation ---

(deftest test-log-gammaln-validates-input
  (testing "log-gammaln signals error for z <= 0"
    (ok (signals (dist:log-gammaln 0.0d0)))
    (ok (signals (dist:log-gammaln -1.0d0)))))

;;; --- HMC validation ---

(deftest test-hmc-validates-inputs
  (let ((dummy-fn (lambda (params)
                    (declare (ignore params))
                    0.0d0)))
    (testing "hmc signals error for empty initial-params"
      (ok (signals (infer:hmc dummy-fn '()))))
    (testing "hmc signals error for n-samples = 0"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-samples 0))))
    (testing "hmc signals error for non-integer n-samples"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-samples 5.5d0))))
    (testing "hmc signals error for negative n-warmup"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-warmup -1))))
    (testing "hmc signals error for non-integer n-warmup"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-warmup 1.5d0))))
    (testing "hmc signals error for non-positive step-size"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :step-size 0.0d0))))
    (testing "hmc signals error for non-positive n-leapfrog"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-leapfrog 0))))
    (testing "hmc signals error for non-integer n-leapfrog"
      (ok (signals (infer:hmc dummy-fn '(0.0d0) :n-leapfrog 5.5d0))))))

;;; --- Optimizer validation ---

(deftest test-sgd-step-validates-lengths
  (testing "sgd-step signals error for mismatched param/grad lengths"
    (ok (signals (opt:sgd-step '(1.0d0) '(1.0d0 2.0d0))))
    (ok (signals (opt:sgd-step '(1.0d0 2.0d0) '(1.0d0))))))

(deftest test-sgd-step-validates-lr
  (testing "sgd-step signals error for non-positive lr"
    (ok (signals (opt:sgd-step '(1.0d0) '(1.0d0) :lr 0)))
    (ok (signals (opt:sgd-step '(1.0d0) '(1.0d0) :lr -0.01d0)))))

(deftest test-adam-step-validates-lengths
  (let ((state (opt:make-adam-state 2)))
    (testing "adam-step signals error for mismatched param/grad lengths"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0 2.0d0) state))))
    (testing "adam-step signals error for param/state dimension mismatch"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state))))))

(deftest test-adam-step-validates-hyperparams
  (let ((state (opt:make-adam-state 1)))
    (testing "adam-step signals error for non-positive lr"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :lr 0)))
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :lr -0.001d0))))
    (testing "adam-step signals error for beta1 outside [0, 1)"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :beta1 1.0d0)))
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :beta1 -0.1d0))))
    (testing "adam-step signals error for beta2 outside [0, 1)"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :beta2 1.0d0)))
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :beta2 -0.1d0))))
    (testing "adam-step signals error for non-positive epsilon"
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :epsilon 0)))
      (ok (signals (opt:adam-step '(1.0d0) '(1.0d0) state :epsilon -1d-8))))))
