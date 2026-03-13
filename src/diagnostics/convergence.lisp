(in-package #:cl-acorn.diagnostics)

;;;; Convergence diagnostics

;;; ---- Internal helpers -----------------------------------------------

(defun chain-param (chains param-idx)
  "Extract parameter PARAM-IDX across all chains.
Returns list of lists: one list per chain of double-float values."
  (mapcar (lambda (chain)
            (mapcar (lambda (sample)
                      (float (nth param-idx sample) 0.0d0))
                    chain))
          chains))

(defun mean-of (xs)
  "Arithmetic mean of a list of numbers."
  (/ (reduce #'+ xs :initial-value 0.0d0) (float (length xs) 0.0d0)))

(defun variance-of (xs)
  "Sample variance (divides by N-1) of a list of numbers."
  (let* ((n (length xs))
         (m (mean-of xs)))
    (if (<= n 1)
        0.0d0
        (/ (reduce (lambda (acc x) (+ acc (expt (- x m) 2)))
                   xs :initial-value 0.0d0)
           (float (1- n) 0.0d0)))))

;;; ---- R-hat (Gelman-Rubin) -------------------------------------------

(defun r-hat-1 (param-chains)
  "Compute R-hat for a single parameter.
PARAM-CHAINS: list of chains, each chain is a list of double-float."
  (let* ((n           (float (length (first param-chains)) 0.0d0))
         (chain-vars  (mapcar #'variance-of param-chains))
         (w           (mean-of chain-vars))
         (chain-means (mapcar #'mean-of param-chains))
         (b           (* n (variance-of chain-means)))
         (v-hat       (+ (* (/ (1- n) n) w) (/ b n))))
    (if (< w 1d-15)
        1.0d0
        (sqrt (/ v-hat w)))))

(defun r-hat (chains)
  "Compute per-parameter R-hat (Gelman-Rubin) from CHAINS.
CHAINS: list of chains; each chain is a list of parameter vectors
\(each parameter vector is a list of numbers).
Returns a list of double-float R-hat values, one per parameter."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (r-hat-1 (chain-param chains i)))))
