(in-package #:cl-acorn.benchmarks)

;;; Target functions for AD benchmarks

(defun sum-squares (params)
  "Compute sum of squares: f(x) = sum(x_i^2). Uses ad: arithmetic."
  (reduce #'ad:+ params :key (lambda (x) (ad:* x x))))

(defun scalar-square (x)
  "Compute x^2 for scalar forward-mode benchmarks."
  (ad:* x x))

;;; Pre-allocated inputs (avoid measuring list allocation in the benchmark)

(defvar *x-1d*    (list 1.0d0))
(defvar *x-10d*   (make-list 10   :initial-element 1.0d0))
(defvar *x-100d*  (make-list 100  :initial-element 1.0d0))
(defvar *x-1000d* (make-list 1000 :initial-element 1.0d0))
(defvar *v-10d*   (make-list 10   :initial-element 1.0d0))

;;; Benchmark runner

(defun run-ad-benchmarks ()
  "Run all AD engine benchmarks. Returns a list of BENCH-RESULT."
  (list
   (defbench "derivative-1d" (:n-runs 10000 :n-warmup 200)
     (ad:derivative #'scalar-square 1.0d0))

   (defbench "gradient-1d" (:n-runs 10000 :n-warmup 200)
     (ad:gradient #'sum-squares *x-1d*))

   (defbench "gradient-10d" (:n-runs 1000 :n-warmup 50)
     (ad:gradient #'sum-squares *x-10d*))

   (defbench "gradient-100d" (:n-runs 500 :n-warmup 20)
     (ad:gradient #'sum-squares *x-100d*))

   (defbench "gradient-1000d" (:n-runs 100 :n-warmup 10)
     (ad:gradient #'sum-squares *x-1000d*))

   (defbench "hessian-vector-product-10d" (:n-runs 500 :n-warmup 20)
     (ad:hessian-vector-product #'sum-squares *x-10d* *v-10d*))))
