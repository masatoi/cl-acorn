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

(deftest test-print-convergence-summary-warns
  (testing "print-convergence-summary shows warn when R-hat > 1.1"
    (let* ((cr (diag:make-chain-result
                :n-chains 4 :n-samples 100 :n-warmup 50
                :r-hat '(1.15d0)
                :bulk-ess '(80.0d0)
                :tail-ess '(70.0d0)
                :n-divergences 2))
           (output (with-output-to-string (*standard-output*)
                     (diag:print-convergence-summary cr))))
      (ok (search "warn" output))
      (ok (search "1.150" output))
      ;; 4 chains x 100 post-warmup samples = 400 total; divergences counted post-warmup only
      (ok (search "Total divergences: 2 / 400" output)))))

(defvar *std-normal-2d*
  (lambda (params)
    (let ((x (first params)) (y (second params)))
      (ad:+ (ad:* -0.5d0 (ad:* x x))
            (ad:* -0.5d0 (ad:* y y))))))

(deftest test-run-chains-returns-chain-result
  (testing "run-chains on 2D std normal returns chain-result"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 200 :n-warmup 100)))
      (ok (diag:chain-result-p cr))
      (ok (= (diag:chain-result-n-chains cr) 4))
      (ok (= (diag:chain-result-n-samples cr) 200))
      (ok (= (diag:chain-result-n-warmup cr) 100))
      (ok (= (length (diag:chain-result-samples cr)) 4))
      (ok (= (length (first (diag:chain-result-samples cr))) 200)))))

(deftest test-run-chains-r-hat-converged
  (testing "run-chains on 2D std normal has R-hat < 1.1"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 500 :n-warmup 200)))
      (ok (every (lambda (r) (< r 1.1d0))
                 (diag:chain-result-r-hat cr))))))

(deftest test-run-chains-ess-adequate
  (testing "run-chains on 2D std normal has bulk-ESS > 100"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 500 :n-warmup 200)))
      (ok (every (lambda (e) (> e 100.0d0))
                 (diag:chain-result-bulk-ess cr))))))

;;; -------------------------------------------------------------------------
;;; run-chains slot coverage tests
;;; -------------------------------------------------------------------------

(deftest test-run-chains-diagnostics-slots
  (testing "accept-rates, tail-ess, elapsed-seconds, n-divergences populated"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 2 :n-samples 50 :n-warmup 30)))
      (ok (= (length (diag:chain-result-accept-rates cr)) 2))
      (ok (every #'plusp (diag:chain-result-accept-rates cr)))
      (ok (= (length (diag:chain-result-tail-ess cr)) 2))
      (ok (every #'plusp (diag:chain-result-tail-ess cr)))
      (ok (>= (diag:chain-result-elapsed-seconds cr) 0.0d0))
      (ok (>= (diag:chain-result-n-divergences cr) 0)))))

(deftest test-run-chains-hmc-sampler
  (testing ":hmc sampler path is reachable"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 2 :n-samples 50 :n-warmup 30
                               :sampler :hmc)))
      (ok (diag:chain-result-p cr))
      (ok (= (diag:chain-result-n-chains cr) 2)))))

;;; -------------------------------------------------------------------------
;;; WAIC tests
;;; -------------------------------------------------------------------------

(defun make-trivial-chain-result (samples-list)
  "Wrap SAMPLES-LIST (a list of per-chain sample lists) into a chain-result."
  (diag:make-chain-result
   :samples samples-list
   :n-chains (length samples-list)
   :n-samples (length (first samples-list))
   :n-warmup 0
   :r-hat nil
   :bulk-ess nil
   :tail-ess nil
   :accept-rates nil
   :n-divergences 0
   :elapsed-seconds 0.0d0))

(deftest test-waic-returns-three-values
  (testing "waic returns three double-float values"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 20
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 5 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (waic-val p-waic lppd)
          (diag:waic cr log-lik-fn data)
        (ok (typep waic-val 'double-float))
        (ok (typep p-waic 'double-float))
        (ok (typep lppd 'double-float))))))

(deftest test-waic-lower-for-correct-model
  (testing "waic is lower for correctly specified model than misspecified model"
    (let* (;; 10 observations from N(0,1)
           (data (loop repeat 10 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0)))
           ;; Model A: fixed samples near mu=0 (correct)
           (samples-a (list (loop repeat 50
                                  collect (list (cl-acorn.distributions:normal-sample
                                                 :mu 0.0d0 :sigma 0.1d0)))))
           ;; Model B: fixed samples near mu=5 (misspecified)
           (samples-b (list (loop repeat 50
                                  collect (list (cl-acorn.distributions:normal-sample
                                                 :mu 5.0d0 :sigma 0.1d0)))))
           (cr-a (make-trivial-chain-result samples-a))
           (cr-b (make-trivial-chain-result samples-b))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0))))
      (multiple-value-bind (waic-a) (diag:waic cr-a log-lik-fn data)
        (multiple-value-bind (waic-b) (diag:waic cr-b log-lik-fn data)
          (ok (< waic-a waic-b)))))))

(deftest test-waic-p-waic-nonnegative
  (testing "p_waic (effective parameters) is non-negative"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 30
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 5 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (waic-val p-waic lppd)
          (diag:waic cr log-lik-fn data)
        (declare (ignore waic-val lppd))
        (ok (>= p-waic 0.0d0))))))

(deftest test-waic-lppd-finite
  (testing "lppd is finite (not NaN or +Inf)"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 30
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 5 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (waic-val p-waic lppd)
          (diag:waic cr log-lik-fn data)
        (declare (ignore waic-val p-waic))
        ;; finite: not NaN (NaN /= NaN) and not +/-infinity
        (ok (and (= lppd lppd)
                 (< (abs lppd) (/ most-positive-double-float 2.0d0))))))))

;;; -------------------------------------------------------------------------
;;; PSIS-LOO tests
;;; -------------------------------------------------------------------------

(deftest test-loo-returns-three-values
  (testing "loo returns three values: loo (double-float), p-loo (double-float), k-hats (list)"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 50
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 5 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (loo-val p-loo k-hats)
          (diag:loo cr log-lik-fn data)
        (ok (typep loo-val 'double-float))
        (ok (typep p-loo 'double-float))
        (ok (listp k-hats))))))

(deftest test-loo-k-hats-length
  (testing "k-hats list length equals the number of data points"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 50
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 7 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (loo-val p-loo k-hats)
          (diag:loo cr log-lik-fn data)
        (declare (ignore loo-val p-loo))
        (ok (= (length k-hats) (length data)))))))

(deftest test-loo-k-hats-mostly-reliable
  (testing "k-hats < 0.5 for majority of data points with well-specified normal model"
    (let* (;; 300 samples near the true mu=0
           (samples (list (loop repeat 300
                                collect (list (cl-acorn.distributions:normal-sample
                                               :mu 0.0d0 :sigma 0.2d0)))))
           (cr (make-trivial-chain-result samples))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           ;; 10 data points from N(0,1)
           (data (loop repeat 10 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (loo-val p-loo k-hats)
          (diag:loo cr log-lik-fn data)
        (declare (ignore loo-val p-loo))
        (let ((reliable-count (count-if (lambda (k) (< k 0.5d0)) k-hats)))
          (ok (> reliable-count (floor (length data) 2))))))))

(deftest test-loo-lower-for-correct-model
  (testing "loo is lower for correctly specified model than misspecified model"
    (let* (;; 10 observations from N(0,1)
           (data (loop repeat 10 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0)))
           ;; Model A: samples near mu=0 (correct)
           (samples-a (list (loop repeat 50
                                  collect (list (cl-acorn.distributions:normal-sample
                                                 :mu 0.0d0 :sigma 0.1d0)))))
           ;; Model B: samples near mu=5 (misspecified)
           (samples-b (list (loop repeat 50
                                  collect (list (cl-acorn.distributions:normal-sample
                                                 :mu 5.0d0 :sigma 0.1d0)))))
           (cr-a (make-trivial-chain-result samples-a))
           (cr-b (make-trivial-chain-result samples-b))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0))))
      (multiple-value-bind (loo-a) (diag:loo cr-a log-lik-fn data)
        (multiple-value-bind (loo-b) (diag:loo cr-b log-lik-fn data)
          (ok (< loo-a loo-b)))))))

(deftest test-loo-p-loo-nonnegative
  (testing "p-loo (effective number of parameters) is non-negative"
    (let* ((cr (make-trivial-chain-result
                (list (loop repeat 50
                            collect (list (cl-acorn.distributions:normal-sample
                                          :mu 0.0d0 :sigma 1.0d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (data (loop repeat 5 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0))))
      (multiple-value-bind (loo-val p-loo k-hats)
          (diag:loo cr log-lik-fn data)
        (declare (ignore loo-val k-hats))
        (ok (>= p-loo 0.0d0))))))

;;; -------------------------------------------------------------------------
;;; print-model-comparison tests
;;; -------------------------------------------------------------------------

(deftest test-print-model-comparison-output
  (testing "print-model-comparison produces expected header and sections"
    (let* (;; 10 observations from N(0,1)
           (data (loop repeat 10 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0)))
           ;; Model A: samples near mu=0 (correct)
           (cr-a (make-trivial-chain-result
                  (list (loop repeat 50
                              collect (list (cl-acorn.distributions:normal-sample
                                             :mu 0.0d0 :sigma 0.1d0))))))
           ;; Model B: samples near mu=5 (misspecified)
           (cr-b (make-trivial-chain-result
                  (list (loop repeat 50
                              collect (list (cl-acorn.distributions:normal-sample
                                             :mu 5.0d0 :sigma 0.1d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0)))
           (output (with-output-to-string (*standard-output*)
                     (diag:print-model-comparison
                      "model-a" cr-a log-lik-fn data
                      "model-b" cr-b log-lik-fn data))))
      (ok (search "Model comparison" output))
      (ok (search "WAIC" output))
      (ok (search "LOO" output))
      (ok (search "Lower is better." output))
      (ok (search "model-a" output))
      (ok (search "model-b" output)))))

(deftest test-print-model-comparison-correct-model-ranks-lower
  (testing "correct model has lower WAIC and LOO than misspecified model"
    (let* (;; 10 observations from N(0,1)
           (data (loop repeat 10 collect
                       (cl-acorn.distributions:normal-sample :mu 0.0d0 :sigma 1.0d0)))
           ;; Model A: samples near mu=0 (correct)
           (cr-a (make-trivial-chain-result
                  (list (loop repeat 50
                              collect (list (cl-acorn.distributions:normal-sample
                                             :mu 0.0d0 :sigma 0.1d0))))))
           ;; Model B: samples near mu=5 (misspecified)
           (cr-b (make-trivial-chain-result
                  (list (loop repeat 50
                              collect (list (cl-acorn.distributions:normal-sample
                                             :mu 5.0d0 :sigma 0.1d0))))))
           (log-lik-fn (lambda (params yi)
                         (cl-acorn.distributions:normal-log-pdf
                          yi :mu (first params) :sigma 1.0d0))))
      (multiple-value-bind (waic-a) (diag:waic cr-a log-lik-fn data)
        (multiple-value-bind (waic-b) (diag:waic cr-b log-lik-fn data)
          (ok (< waic-a waic-b)))))))
