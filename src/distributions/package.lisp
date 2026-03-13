(defpackage #:cl-acorn.distributions
  (:nicknames #:dist)
  (:use #:cl)
  (:export
   ;; Utilities
   #:log-gammaln
   #:+log-pdf-sentinel+
   ;; Normal
   #:normal-log-pdf #:normal-sample
   ;; Gamma
   #:gamma-log-pdf #:gamma-sample
   ;; Beta
   #:beta-log-pdf #:beta-sample
   ;; Uniform
   #:uniform-log-pdf #:uniform-sample
   ;; Bernoulli
   #:bernoulli-log-pdf #:bernoulli-sample
   ;; Poisson
   #:poisson-log-pdf #:poisson-sample))
