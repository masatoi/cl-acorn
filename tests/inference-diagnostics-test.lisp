(defpackage #:cl-acorn/tests/inference-diagnostics-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/inference-diagnostics-test)

(defvar *std-normal*
  (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))

(deftest test-hmc-returns-diagnostics
  (testing "hmc returns inference-diagnostics as 3rd value"
    (multiple-value-bind (samples ar diag)
        (infer:hmc *std-normal* '(0.0d0) :n-samples 100 :n-warmup 50)
      (declare (ignore samples ar))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 50))
      (ok (>= (infer:diagnostics-accept-rate diag) 0.0d0))
      (ok (<= (infer:diagnostics-accept-rate diag) 1.0d0))
      (ok (= (infer:diagnostics-n-divergences diag) 0))
      (ok (>= (infer:diagnostics-elapsed-seconds diag) 0.0d0)))))

(deftest test-hmc-diagnostics-step-size-set
  (testing "hmc diagnostics records final step-size"
    (multiple-value-bind (samples ar diag)
        (infer:hmc *std-normal* '(0.0d0)
                   :n-samples 100 :n-warmup 50
                   :step-size 0.2d0 :adapt-step-size nil)
      (declare (ignore samples ar))
      (ok (> (infer:diagnostics-final-step-size diag) 0.0d0)))))

(deftest test-nuts-returns-diagnostics
  (testing "nuts returns inference-diagnostics as 3rd value"
    (multiple-value-bind (samples ar diag)
        (infer:nuts *std-normal* '(0.0d0) :n-samples 100 :n-warmup 50)
      (declare (ignore samples ar))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 50))
      (ok (>= (infer:diagnostics-n-divergences diag) 0)))))

(deftest test-nuts-diagnostics-counts-divergences
  (testing "nuts diagnostics counts post-warmup divergences"
    (handler-bind ((infer:high-divergence-warning
                     (lambda (c)
                       (declare (ignore c))
                       (invoke-restart 'infer:continue-with-warnings))))
      (multiple-value-bind (samples ar diag)
          (infer:nuts *std-normal* '(0.0d0)
                      :n-samples 50 :n-warmup 5
                      :step-size 100.0d0 :adapt-step-size nil)
        (declare (ignore samples ar))
        (ok (> (infer:diagnostics-n-divergences diag) 0))))))

(deftest test-vi-returns-diagnostics
  (testing "vi returns inference-diagnostics as 4th value"
    (multiple-value-bind (mu sigma elbo diag)
        (infer:vi *std-normal* 1 :n-iterations 100 :n-elbo-samples 5)
      (declare (ignore mu sigma elbo))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 0))
      (ok (>= (infer:diagnostics-elapsed-seconds diag) 0.0d0)))))

(deftest test-hmc-backward-compatible
  (testing "existing 2-value multiple-value-bind still works"
    (multiple-value-bind (samples ar)
        (infer:hmc *std-normal* '(0.0d0) :n-samples 50 :n-warmup 20)
      (ok (> (length samples) 0))
      (ok (> ar 0.0d0)))))

(deftest test-nuts-backward-compatible
  (testing "existing 2-value multiple-value-bind for nuts still works"
    (multiple-value-bind (samples ar)
        (infer:nuts *std-normal* '(0.0d0) :n-samples 50 :n-warmup 20)
      (ok (> (length samples) 0))
      (ok (>= ar 0.0d0)))))
