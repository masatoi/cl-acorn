(in-package #:cl-acorn.diagnostics)

;;;; chain-result struct

(defstruct chain-result
  "Aggregated results from a multi-chain MCMC run."
  (samples         nil)
  (n-chains        0   :type (integer 0))
  (n-samples       0   :type (integer 0))
  (n-warmup        0   :type (integer 0))
  (r-hat           nil)
  (bulk-ess        nil)
  (tail-ess        nil)
  (accept-rates    nil)
  (n-divergences   0   :type (integer 0))
  (elapsed-seconds 0.0d0 :type double-float))

;;;; run-chains

;;; ---- Helper: perturb initial params -----------------------------------

(defun jitter-params (params)
  "Add N(0, 0.1) noise to each element of PARAMS.
Returns a new list of double-float values."
  (mapcar (lambda (p)
            (cl-acorn.distributions:normal-sample
             :mu    (float p 0.0d0)
             :sigma 0.1d0))
          params))

;;; ---- Main entry point -------------------------------------------------

(defun run-chains (log-pdf-fn initial-params
                   &key (n-chains      4)
                        (n-samples  1000)
                        (n-warmup    500)
                        (sampler    :nuts)
                        (adapt-step-size t)
                        (step-size  0.1d0))
  "Run N-CHAINS independent MCMC chains and return a CHAIN-RESULT.

Each chain starts from INITIAL-PARAMS perturbed by N(0, 0.1) jitter to
ensure chains explore from different starting points. Chains run sequentially.
R-hat, bulk-ESS, and tail-ESS are automatically computed from the
post-warmup samples and stored in the returned CHAIN-RESULT.

LOG-PDF-FN: (lambda (params) -> number) using ad: arithmetic
INITIAL-PARAMS: list of initial parameter values (numbers)
SAMPLER: :nuts (default) or :hmc
ADAPT-STEP-SIZE: whether to adapt step size during warmup (default t)
STEP-SIZE: initial step size (default 0.1)

Returns a CHAIN-RESULT struct."
  (let* ((t0             (get-internal-real-time))
         (all-samples    (make-list n-chains))
         (all-accept     (make-list n-chains))
         (total-div      0))
    (loop for i from 0 below n-chains do
      (let* ((start (jitter-params initial-params))
             (runner
               (ecase sampler
                 (:nuts (lambda ()
                          (cl-acorn.inference:nuts
                           log-pdf-fn start
                           :n-samples n-samples
                           :n-warmup n-warmup
                           :step-size step-size
                           :adapt-step-size adapt-step-size)))
                 (:hmc  (lambda ()
                          (cl-acorn.inference:hmc
                           log-pdf-fn start
                           :n-samples n-samples
                           :n-warmup n-warmup
                           :step-size step-size
                           :adapt-step-size adapt-step-size))))))
        (multiple-value-bind (samples accept-rate diag)
            (funcall runner)
          (setf (nth i all-samples) samples)
          (setf (nth i all-accept)  (float accept-rate 0.0d0))
          (incf total-div
                (cl-acorn.inference:diagnostics-n-divergences diag)))))
    (let* ((elapsed (/ (float (- (get-internal-real-time) t0) 0.0d0)
                       (float internal-time-units-per-second 0.0d0)))
           (rhat    (r-hat  all-samples))
           (bess    (bulk-ess all-samples))
           (tess    (tail-ess all-samples)))
      (make-chain-result
       :samples          all-samples
       :n-chains         n-chains
       :n-samples        n-samples
       :n-warmup         n-warmup
       :r-hat            rhat
       :bulk-ess         bess
       :tail-ess         tess
       :accept-rates     all-accept
       :n-divergences    total-div
       :elapsed-seconds  (float elapsed 0.0d0)))))
