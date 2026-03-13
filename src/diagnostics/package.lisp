(defpackage #:cl-acorn.diagnostics
  (:nicknames #:diag)
  (:use #:cl)
  (:export
   ;; chains.lisp
   #:run-chains
   #:chain-result #:chain-result-p
   #:chain-result-samples #:chain-result-n-chains
   #:chain-result-n-samples #:chain-result-n-warmup
   #:chain-result-r-hat #:chain-result-bulk-ess #:chain-result-tail-ess
   #:chain-result-accept-rates #:chain-result-n-divergences
   #:chain-result-elapsed-seconds
   ;; convergence.lisp
   #:r-hat #:bulk-ess #:tail-ess
   #:print-convergence-summary
   ;; model-comparison.lisp
   #:waic #:loo #:print-model-comparison))

(in-package #:cl-acorn.diagnostics)
