(in-package #:cl-acorn.benchmarks)

;;; 2D standard normal: log p(x1,x2) = -0.5*(x1^2 + x2^2)

(defun log-normal-2d (params)
  "Log-density of 2D standard normal. Uses ad: arithmetic for gradient computation."
  (let ((x1 (first params))
        (x2 (second params)))
    (ad:* -0.5d0 (ad:+ (ad:* x1 x1) (ad:* x2 x2)))))

(defparameter +bench-n-samples+ 500
  "Number of posterior samples for inference benchmarks.")

(defparameter +bench-n-warmup+ 200
  "Number of warmup iterations for inference benchmarks.")

(defun bench-hmc ()
  "Benchmark HMC on 2D standard normal.
Returns a BENCH-RESULT where mean-us = microseconds per sample."
  (multiple-value-bind (samples ar diag)
      (infer:hmc #'log-normal-2d '(0.0d0 0.0d0)
                 :n-samples +bench-n-samples+
                 :n-warmup  +bench-n-warmup+
                 :adapt-step-size t)
    (declare (ignore samples ar))
    (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
           (mean-us (* 1.0d6 (/ elapsed (float +bench-n-samples+ 0.0d0)))))
      (make-bench-result
       :name     "hmc-2d-standard-normal"
       :mean-us  mean-us
       :min-us   mean-us
       :max-us   mean-us
       :gc-bytes 0))))

(defun bench-nuts ()
  "Benchmark NUTS on 2D standard normal.
Returns a BENCH-RESULT where mean-us = microseconds per sample."
  (multiple-value-bind (samples ar diag)
      (infer:nuts #'log-normal-2d '(0.0d0 0.0d0)
                  :n-samples +bench-n-samples+
                  :n-warmup  +bench-n-warmup+
                  :adapt-step-size t)
    (declare (ignore samples ar))
    (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
           (mean-us (* 1.0d6 (/ elapsed (float +bench-n-samples+ 0.0d0)))))
      (make-bench-result
       :name     "nuts-2d-standard-normal"
       :mean-us  mean-us
       :min-us   mean-us
       :max-us   mean-us
       :gc-bytes 0))))

(defun bench-vi ()
  "Benchmark VI (ADVI) on 2D standard normal.
Returns a BENCH-RESULT where mean-us = microseconds per iteration."
  (let ((n-iterations 1000))
    (multiple-value-bind (mu sigma elbo diag)
        (infer:vi #'log-normal-2d 2
                  :n-iterations   n-iterations
                  :n-elbo-samples 10
                  :lr 0.01d0)
      (declare (ignore mu sigma elbo))
      (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
             (mean-us (* 1.0d6 (/ elapsed (float n-iterations 0.0d0)))))
        (make-bench-result
         :name     "vi-2d-standard-normal"
         :mean-us  mean-us
         :min-us   mean-us
         :max-us   mean-us
         :gc-bytes 0)))))

(defun run-inference-benchmarks ()
  "Run all inference benchmarks (HMC, NUTS, VI) on 2D standard normal.
Each result's mean-us = microseconds per sample (HMC/NUTS) or per iteration (VI)."
  (list (bench-hmc) (bench-nuts) (bench-vi)))
