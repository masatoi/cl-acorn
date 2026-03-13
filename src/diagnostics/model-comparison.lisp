(in-package #:cl-acorn.diagnostics)

;;; -------------------------------------------------------------------------
;;; Internal helpers
;;; -------------------------------------------------------------------------

(defun all-chain-samples (chain-result)
  "Return a flat list of all post-warmup parameter vectors from CHAIN-RESULT."
  (apply #'append (chain-result-samples chain-result)))

(defun log-sum-exp (log-vals)
  "Numerically stable log(sum(exp(x))) for a list LOG-VALS of log-values."
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
         (s (length samples))
         (lppd 0.0d0)
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
            (float lppd   0.0d0))))
