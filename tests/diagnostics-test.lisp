(defpackage #:cl-acorn/tests/diagnostics-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/diagnostics-test)

(deftest test-chain-result-struct
  (testing "chain-result is a struct with expected accessors"
    (let ((cr (diag:make-chain-result
               :samples '(((1.0d0 2.0d0)) ((1.1d0 2.1d0)))
               :n-chains 2
               :n-samples 1
               :n-warmup 0
               :r-hat '(1.0d0 1.0d0)
               :bulk-ess '(1.0d0 1.0d0)
               :tail-ess '(1.0d0 1.0d0)
               :accept-rates '(0.9d0 0.8d0)
               :n-divergences 0
               :elapsed-seconds 0.01d0)))
      (ok (diag:chain-result-p cr))
      (ok (= (diag:chain-result-n-chains cr) 2))
      (ok (= (diag:chain-result-n-samples cr) 1))
      (ok (= (diag:chain-result-n-warmup cr) 0))
      (ok (= (diag:chain-result-n-divergences cr) 0))
      (ok (approx= (diag:chain-result-elapsed-seconds cr) 0.01d0))
      (ok (equal (diag:chain-result-samples cr) '(((1.0d0 2.0d0)) ((1.1d0 2.1d0)))))
      (ok (equal (diag:chain-result-r-hat cr) '(1.0d0 1.0d0)))
      (ok (equal (diag:chain-result-accept-rates cr) '(0.9d0 0.8d0))))))
