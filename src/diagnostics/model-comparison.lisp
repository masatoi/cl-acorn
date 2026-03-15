(in-package #:cl-acorn.diagnostics)

;;; -------------------------------------------------------------------------
;;; Internal helpers
;;; -------------------------------------------------------------------------

(defun all-chain-samples (chain-result)
  "Return a flat list of all post-warmup parameter vectors from CHAIN-RESULT."
  (apply #'append (chain-result-samples chain-result)))

(defun log-sum-exp (log-vals)
  "Numerically stable log(sum(exp(x))) for a list LOG-VALS of log-values.
Returns MOST-NEGATIVE-DOUBLE-FLOAT for an empty list."
  (when (null log-vals)
    (return-from log-sum-exp most-negative-double-float))
  (let* ((m (reduce #'max log-vals))
         (sum (reduce #'+ log-vals :key (lambda (x) (exp (- x m))))))
    (+ m (log sum))))

;;; -------------------------------------------------------------------------
;;; WAIC — Widely Applicable Information Criterion
;;; -------------------------------------------------------------------------

(defun waic (chain-result log-likelihood-fn data)
  "Compute WAIC (Widely Applicable Information Criterion) from posterior samples.

CHAIN-RESULT: output of RUN-CHAINS
LOG-LIKELIHOOD-FN: (lambda (params data-point) -> double-float)
DATA: sequence of data points (list or vector)

Returns (values waic p-waic lppd) where:
  WAIC   = -2 * (lppd - p_waic)  -- lower is better
  P-WAIC = effective number of parameters (penalty term)
  LPPD   = log pointwise predictive density"
  (let* ((samples (all-chain-samples chain-result))
         (s (length samples)))
    (when (< s 2)
      (error "~A requires at least 2 posterior samples; got ~D" 'waic s))
    (let ((lppd 0.0d0)
          (p-waic 0.0d0))
      (map nil
           (lambda (yi)
             (let* ((log-liks
                      (mapcar (lambda (theta)
                                (float (funcall log-likelihood-fn theta yi) 0.0d0))
                              samples))
                    ;; lppd contribution: log(mean_s p(y_i|theta_s))
                    ;;   = log-sum-exp(log_liks) - log(S)
                    (lppd-i (- (log-sum-exp log-liks) (log (float s 0.0d0))))
                    ;; p_waic contribution: var_s(log p(y_i|theta_s))
                    (mean-ll (/ (reduce #'+ log-liks) (float s 0.0d0)))
                    (p-waic-i (/ (reduce #'+ log-liks
                                         :key (lambda (x) (expt (- x mean-ll) 2)))
                                 (float (1- s) 0.0d0))))
               (incf lppd   lppd-i)
               (incf p-waic p-waic-i)))
           data)
      (values (float (* -2.0d0 (- lppd p-waic)) 0.0d0)
              (float p-waic 0.0d0)
              (float lppd   0.0d0)))))

;;; -------------------------------------------------------------------------
;;; PSIS-LOO — Pareto-Smoothed Importance Sampling LOO Cross-Validation
;;; -------------------------------------------------------------------------

(defun fit-pareto-shape (tail-weights)
  "Estimate Pareto shape k from sorted tail weights using the Zhang-Stephens moment estimator.
TAIL-WEIGHTS: list of positive numbers (the M largest raw weights, sorted ascending).
Zero or negative weights are filtered out before fitting.
Returns k-hat as double-float."
  (let* ((pos-weights (remove-if (lambda (w) (<= w 0.0d0)) tail-weights))
         (n (length pos-weights)))
    (if (< n 5)
        0.0d0
        (let ((log-z1 (log (first pos-weights))))  ; guaranteed positive
          (/ (reduce #'+ pos-weights
                     :key (lambda (z) (- (log (max z 1d-300)) log-z1)))
             (float n 0.0d0))))))

(defun psis-smooth-weights (log-weights)
  "Apply PSIS smoothing to LOG-WEIGHTS for a single observation.
Returns (values normalized-weights k-hat) where:
  NORMALIZED-WEIGHTS: list of normalized importance weights (sum to 1)
  K-HAT: Pareto shape parameter (k < 0.5 reliable, 0.5-0.7 acceptable, >= 0.7 unreliable)"
  (let* ((s (length log-weights)))
    (when (< s 5)
      (return-from psis-smooth-weights
        (values (make-list s :initial-element (/ 1.0d0 (max s 1))) 0.0d0)))
    (let* ((m (max 5 (min (floor (/ s 5))
                          (floor (* 3.0d0 (sqrt (float s 0.0d0)))))))
           (max-lw (reduce #'max log-weights))
           (raw-w (map 'vector (lambda (lw) (exp (- lw max-lw))) log-weights))
           ;; sorted copy for tail extraction
           (sorted-w (sort (copy-seq raw-w) #'<))
           ;; smallest value in the top-M tail
           (tail-threshold (aref sorted-w (- s m)))
           ;; top-M values sorted ascending for Pareto fit
           (tail-vals (coerce (subseq sorted-w (- s m)) 'list))
           (k-hat (fit-pareto-shape tail-vals))
           (smoothed-w (copy-seq raw-w)))
      (when (> k-hat 0.0d0)
        (let* ((tail-mean (/ (reduce #'+ tail-vals) (float m 0.0d0)))
               (sigma (max 1d-300 (* tail-mean k-hat)))
               (sorted-tail-indices
                 ;; Collect all indices at or above the tail threshold,
                 ;; sort ascending by weight, then cap to exactly m entries.
                 ;; Capping prevents tied boundary values from pushing rank > m,
                 ;; which would make q = (rank-0.5)/m > 1 and yield z < 0.
                 (let ((tail-indices
                         (loop for i from 0 below s
                               when (>= (aref raw-w i) tail-threshold)
                               collect i)))
                   (subseq (sort tail-indices #'< :key (lambda (i) (aref raw-w i)))
                           0 (min m (length tail-indices))))))
          (loop for rank from 1
                for idx in sorted-tail-indices
                do (let* ((q (/ (- (float rank 0.0d0) 0.5d0) (float m 0.0d0)))
                           (z (/ (* sigma (- (expt q (- k-hat)) 1.0d0)) k-hat)))
                     (setf (aref smoothed-w idx)
                           (min z (aref sorted-w (1- s))))))))
      (let* ((total (reduce #'+ smoothed-w))
             (denom (max total 1d-300)))
        (values (map 'list (lambda (w) (/ w denom)) smoothed-w)
                k-hat)))))

(defun loo (chain-result log-likelihood-fn data)
  "Compute PSIS-LOO (Pareto-Smoothed Importance Sampling LOO cross-validation).

CHAIN-RESULT: output of RUN-CHAINS
LOG-LIKELIHOOD-FN: (lambda (params data-point) -> double-float)
DATA: sequence of data points (list or vector)

Returns (values loo p-loo k-hats) where:
  LOO    = -2 * loo-lpd  -- lower is better
  P-LOO  = effective number of parameters (lppd - loo-lpd)
  K-HATS = list of per-data-point Pareto shape parameters"
  (let* ((samples (all-chain-samples chain-result))
         (s (length samples)))
    (when (< s 2)
      (error "~A requires at least 2 posterior samples; got ~D" 'loo s))
    (let ((lppd 0.0d0)
          (loo-lpd 0.0d0)
          (k-hats '()))
      (map nil
           (lambda (yi)
             (let* ((log-liks
                      (mapcar (lambda (theta)
                                (float (funcall log-likelihood-fn theta yi) 0.0d0))
                              samples))
                    ;; lppd contribution: log(mean_s p(y_i|theta_s))
                    (lppd-i (- (log-sum-exp log-liks) (log (float s 0.0d0))))
                    ;; LOO importance weights: -log p(y_i | theta_s)
                    (log-iw (mapcar #'- log-liks)))
               (multiple-value-bind (norm-w k-hat)
                   (psis-smooth-weights log-iw)
                 ;; LOO predictive density: log(sum_s w_s * p(y_i|theta_s))
                 (let* ((weighted-liks
                          (mapcar (lambda (w ll) (* w (exp ll))) norm-w log-liks))
                        (loo-i (log (max (reduce #'+ weighted-liks) 1d-300))))
                   (incf lppd    lppd-i)
                   (incf loo-lpd loo-i)
                   (push k-hat k-hats)))))
           data)
      (values (float (* -2.0d0 loo-lpd) 0.0d0)
              (float (- lppd loo-lpd) 0.0d0)
              (nreverse k-hats)))))

;;; -------------------------------------------------------------------------
;;; Model comparison table
;;; -------------------------------------------------------------------------

(defun print-model-comparison (&rest named-results)
  "Print WAIC and LOO comparison table to *STANDARD-OUTPUT*.

NAMED-RESULTS: alternating name/chain-result/log-lik-fn/data groups.
Each group is: NAME (string) CHAIN-RESULT LOG-LIK-FN DATA.

Lower WAIC/LOO = better predictive accuracy."
  (assert (zerop (mod (length named-results) 4)) nil
          "print-model-comparison: NAMED-RESULTS must be groups of 4 (name result log-lik-fn data)")
  (let ((rows
          (loop for i from 0 below (length named-results) by 4
                collect
                (let ((name   (nth i       named-results))
                      (result (nth (+ i 1) named-results))
                      (lik-fn (nth (+ i 2) named-results))
                      (data   (nth (+ i 3) named-results)))
                  (multiple-value-bind (waic-val p-waic lppd-w)
                      (waic result lik-fn data)
                    (declare (ignore lppd-w))
                    (multiple-value-bind (loo-val p-loo k-hats)
                        (loo result lik-fn data)
                      (declare (ignore k-hats))
                      (list name waic-val p-waic loo-val p-loo)))))))
    (format t "Model comparison~%")
    (format t "~55,,,'=A~%" "")
    (format t "~15A | ~8A | ~6A | ~8A | ~5A~%"
            "Model" "WAIC" "p_waic" "LOO" "p_loo")
    (format t "~15,,,'-A-+-~8,,,'-A-+-~6,,,'-A-+-~8,,,'-A-+-~5,,,'-A~%"
            "" "" "" "" "")
    (dolist (row rows)
      (destructuring-bind (name waic-val p-waic loo-val p-loo) row
        (format t "~15A | ~8,1F | ~6,1F | ~8,1F | ~5,1F~%"
                name waic-val p-waic loo-val p-loo)))
    (format t "~55,,,'=A~%" "")
    (format t "Lower is better.~%")))
