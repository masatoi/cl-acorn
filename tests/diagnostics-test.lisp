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

(deftest test-r-hat-converged
  (testing "r-hat near 1.0 for 4 chains from same distribution"
    ;; 4 chains x 200 samples from N(0,1) — all should converge
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (rhat (diag:r-hat chains)))
      (ok (listp rhat))
      (ok (= (length rhat) 1))
      (ok (< (first rhat) 1.1d0)))))

(deftest test-r-hat-not-converged
  (testing "r-hat > 1.1 for chains started far apart with few samples"
    (let* ((chains (list
                    (loop repeat 10 collect (list (+ -10.0d0 (* 0.1d0 (- (random 10) 5)))))
                    (loop repeat 10 collect (list (+  10.0d0 (* 0.1d0 (- (random 10) 5)))))
                    (loop repeat 10 collect (list (+ -10.0d0 (* 0.1d0 (- (random 10) 5)))))
                    (loop repeat 10 collect (list (+  10.0d0 (* 0.1d0 (- (random 10) 5)))))))
           (rhat (diag:r-hat chains)))
      (ok (> (first rhat) 1.1d0)))))

(deftest test-bulk-ess-near-independent
  (testing "bulk-ess > 10% of M*N for near-independent samples"
    ;; 4 chains x 200 samples iid N(0,1) → ESS close to 800
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (ess (diag:bulk-ess chains)))
      (ok (listp ess))
      (ok (= (length ess) 1))
      (ok (> (first ess) 80.0d0)))))   ; at least 10% of 800

(deftest test-tail-ess-near-independent
  (testing "tail-ess > 10% of M*N for near-independent samples"
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (tess (diag:tail-ess chains)))
      (ok (listp tess))
      (ok (= (length tess) 1))
      (ok (> (first tess) 40.0d0)))))  ; tail ESS can be lower

(deftest test-print-convergence-summary-output
  (testing "print-convergence-summary produces expected output"
    (let* ((cr (diag:make-chain-result
                :n-chains 4 :n-samples 100 :n-warmup 50
                :r-hat '(1.001d0 0.999d0)
                :bulk-ess '(923.4d0 887.2d0)
                :tail-ess '(880.0d0 799.6d0)
                :n-divergences 0))
           (output (with-output-to-string (*standard-output*)
                     (diag:print-convergence-summary cr))))
      (ok (search "R-hat" output))
      (ok (search "Bulk-ESS" output))
      (ok (search "1.001" output))
      (ok (search "923" output))
      (ok (search "Total divergences" output)))))
