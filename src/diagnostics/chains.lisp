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
