(defpackage #:cl-acorn.optimizers
  (:nicknames #:opt)
  (:use #:cl)
  (:export
   #:sgd-step
   #:adam-state
   #:make-adam-state
   #:adam-step))
