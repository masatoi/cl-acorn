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
  (let ((n (length xs))
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

;;; ---- Bulk-ESS via autocorrelation -----------------------------------

(defun autocorrelation (xs lag)
  "Normalized autocorrelation of XS at LAG. Returns value in [-1, 1]."
  (let ((n (length xs))
        (m (mean-of xs))
        (v (variance-of xs)))
    (if (< v 1d-15)
        0.0d0
        (let ((cov (loop for i from 0 below (- n lag)
                         sum (* (- (nth i xs) m)
                                (- (nth (+ i lag) xs) m)))))
          (/ cov (* (float (1- n) 0.0d0) v))))))

(defun bulk-ess-1 (param-chains)
  "Compute bulk ESS for a single parameter.
PARAM-CHAINS: list of chains, each a list of double-float."
  (let ((mn          (float (* (length param-chains)
                               (length (first param-chains))) 0.0d0))
        (all-samples (apply #'append param-chains))
        (rho-sum     0.0d0))
    ;; Sum autocorrelations until they go negative (Geyer's initial monotone rule)
    (loop for lag from 1 to (min 100 (1- (length all-samples)))
          for rho = (autocorrelation all-samples lag)
          while (> rho 0.0d0)
          do (incf rho-sum rho))
    (max 1.0d0 (/ mn (+ 1.0d0 (* 2.0d0 rho-sum))))))

(defun bulk-ess (chains)
  "Per-parameter bulk effective sample size.
CHAINS: list of chains; each chain is a list of parameter vectors.
Returns a list of double-float ESS values."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (bulk-ess-1 (chain-param chains i)))))

;;; ---- Tail-ESS via quantile indicators --------------------------------

(defun indicator-ess (param-chains threshold)
  "ESS of the indicator I(x <= THRESHOLD) across PARAM-CHAINS."
  (let ((indicator-chains
          (mapcar (lambda (chain)
                    (mapcar (lambda (x) (if (<= x threshold) 1.0d0 0.0d0))
                            chain))
                  param-chains)))
    (bulk-ess-1 indicator-chains)))

(defun quantile (xs q)
  "Compute quantile Q (in [0,1]) of list XS. Returns a value from XS."
  (let* ((sorted (sort (copy-list xs) #'<))
         (n      (length sorted))
         (idx    (min (1- n) (floor (* q (float n 0.0d0))))))
    (nth idx sorted)))

(defun tail-ess-1 (param-chains)
  "Compute tail ESS for a single parameter.
Uses ESS of I(x<=Q25) and I(x<=Q75); returns their minimum."
  (let* ((all-samples (apply #'append param-chains))
         (q25 (quantile all-samples 0.25d0))
         (q75 (quantile all-samples 0.75d0)))
    (min (indicator-ess param-chains q25)
         (indicator-ess param-chains q75))))

(defun tail-ess (chains)
  "Per-parameter tail effective sample size using Q0.25 and Q0.75 indicators.
Returns the minimum of ESS(I(x<=Q0.25)) and ESS(I(x<=Q0.75)).
Note: canonical Vehtari (2021) uses Q0.05/Q0.95 instead.
CHAINS: list of chains; each chain is a list of parameter vectors.
Returns a list of double-float tail-ESS values."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (tail-ess-1 (chain-param chains i)))))

;;; ---- Convergence summary table --------------------------------------

(defun print-convergence-summary (chain-result)
  "Print convergence diagnostics table to *STANDARD-OUTPUT*.

Output format:
  Convergence diagnostics  (4 chains x 1000 samples)
  ====================================================
  Param | R-hat  | Bulk-ESS | Tail-ESS | Status
  ------|--------|----------|----------|--------
    [0] | 1.001  |    923.4 |    887.2 | ok
  ----
  Total divergences: 0 / 4000

Status is 'ok' when R-hat < 1.1 and Bulk-ESS > 100, else 'warn'."
  (let* ((n-chains  (chain-result-n-chains chain-result))
         (n-samples (chain-result-n-samples chain-result))
         (rhats     (chain-result-r-hat chain-result))
         (bess      (chain-result-bulk-ess chain-result))
         (tess      (chain-result-tail-ess chain-result))
         (ndiv      (chain-result-n-divergences chain-result))
         (total     (* n-chains n-samples)))
    (format t "~%Convergence diagnostics  (~D chains x ~D samples)~%" n-chains n-samples)
    (format t "==============================================~%")
    (format t "~6A | ~6A | ~8A | ~8A | ~6A~%"
            "Param" "R-hat" "Bulk-ESS" "Tail-ESS" "Status")
    (format t "~6A-+-~6A-+-~8A-+-~8A-+-~6A~%"
            "------" "------" "--------" "--------" "------")
    (loop for i from 0
          for rhat in rhats
          for be   in bess
          for te   in tess
          do (let ((status (if (and (< rhat 1.1d0) (> be 100.0d0)) "ok" "warn")))
               (format t "~6A | ~6,3F | ~8,1F | ~8,1F | ~6A~%"
                       (format nil "[~D]" i) rhat be te status)))
    (format t "----------------------------------------------~%")
    (format t "Total divergences: ~D / ~D~%" ndiv total)))
