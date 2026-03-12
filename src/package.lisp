(defpackage #:cl-acorn.ad
  (:nicknames #:ad)
  (:use #:cl)
  (:shadow #:+ #:- #:* #:/
           #:sin #:cos #:tan #:exp #:log #:expt #:sqrt #:abs)
  (:export
   ;; Dual number class and accessors
   #:dual
   #:make-dual
   #:dual-real
   #:dual-epsilon
   ;; Arithmetic
   #:+ #:- #:* #:/
   ;; Transcendental
   #:sin #:cos #:tan #:exp #:log #:expt #:sqrt #:abs
   ;; Interface (forward-mode)
   #:derivative
   ;; Tape node class and accessors (reverse-mode)
   #:tape-node
   #:node-value
   #:node-gradient
   ;; Interface (reverse-mode)
   #:gradient
   #:jacobian-vector-product
   #:hessian-vector-product))
