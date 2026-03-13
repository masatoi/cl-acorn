(in-package #:cl-acorn.benchmarks)

(defun run-distributions-benchmarks ()
  "Run all distribution log-pdf benchmarks. Returns a list of BENCH-RESULT."
  (list
   (defbench "normal-log-pdf" (:n-runs 10000 :n-warmup 200)
     (dist:normal-log-pdf 0.5d0 :mu 0.0d0 :sigma 1.0d0))

   (defbench "gamma-log-pdf" (:n-runs 10000 :n-warmup 200)
     (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 1.0d0))

   (defbench "beta-log-pdf" (:n-runs 10000 :n-warmup 200)
     (dist:beta-log-pdf 0.5d0 :alpha 2.0d0 :beta 3.0d0))

   (defbench "poisson-log-pdf" (:n-runs 10000 :n-warmup 200)
     (dist:poisson-log-pdf 3 :rate 2.0d0))))
