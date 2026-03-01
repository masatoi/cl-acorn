(defpackage #:cl-acorn.ad
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
   ;; Interface
   #:derivative))
