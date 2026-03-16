(in-package #:cl-acorn.inference)

(declaim (inline finite-double-p))
(defun finite-double-p (x)
  "Return T if X can be represented as a finite double-float."
  (handler-case
      (let ((value (coerce x 'double-float)))
        (and (= value value)
             (<= (- most-positive-double-float)
                 value
                 most-positive-double-float)))
    (error () nil)))

(defun every-finite-p (list)
  "Return T if every element in LIST is a finite double-float."
  (every #'finite-double-p list))

(defmacro with-float-traps-masked (&body body)
  "Execute BODY with floating-point traps masked when supported by the runtime."
  #+sbcl
  `(sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
     ,@body)
  #-sbcl
  `(progn ,@body))

(defvar *log-pdf-error-warned-p* nil
  "When non-nil, suppress repeated warnings about non-arithmetic log-pdf errors.
Bound to NIL at the start of each top-level inference call.")

(defconstant +high-divergence-threshold+ 0.10d0
  "Warn when post-warmup divergence rate exceeds this fraction.")

(defun warn-log-pdf-error-once (context condition)
  "Warn once per inference call about a non-arithmetic CONDITION."
  (unless *log-pdf-error-warned-p*
    (warn "~A: caught ~A in log-pdf-fn: ~A~%~
           Further non-arithmetic errors will be suppressed for this call."
          context
          (type-of condition)
          condition)
    (setf *log-pdf-error-warned-p* t)))

(defun ensure-valid-parameter (condition parameter value message)
  "Signal INVALID-PARAMETER-ERROR unless CONDITION is true."
  (unless condition
    (error 'invalid-parameter-error
           :parameter parameter
           :value value
           :message message)))

(defun validate-initial-params (caller initial-params)
  "Validate and coerce INITIAL-PARAMS for CALLER."
  (ensure-valid-parameter
   (and (listp initial-params) (consp initial-params))
   :initial-params
   initial-params
   (format nil "~A: INITIAL-PARAMS must be a non-empty list" caller))
  (mapcar (lambda (x) (coerce x 'double-float)) initial-params))

(defun validate-positive-integer-parameter (caller parameter value)
  "Validate VALUE as a positive integer parameter for CALLER."
  (ensure-valid-parameter
   (and (integerp value) (plusp value))
   parameter
   value
   (format nil "~A: ~A must be a positive integer"
           caller
           (string-upcase (symbol-name parameter)))))

(defun validate-non-negative-integer-parameter (caller parameter value)
  "Validate VALUE as a non-negative integer parameter for CALLER."
  (ensure-valid-parameter
   (and (integerp value) (not (minusp value)))
   parameter
   value
   (format nil "~A: ~A must be a non-negative integer"
           caller
           (string-upcase (symbol-name parameter)))))

(defun validate-positive-real-parameter (caller parameter value)
  "Validate VALUE as a positive real parameter for CALLER and return a double-float."
  (ensure-valid-parameter
   (> value 0.0d0)
   parameter
   value
   (format nil "~A: ~A must be a positive number"
           caller
           (string-upcase (symbol-name parameter))))
  (coerce value 'double-float))

(defun validate-mcmc-common-args (caller initial-params n-samples n-warmup step-size)
  "Validate and coerce the shared top-level MCMC arguments for CALLER."
  (let ((params (validate-initial-params caller initial-params)))
    (validate-positive-integer-parameter caller :n-samples n-samples)
    (validate-non-negative-integer-parameter caller :n-warmup n-warmup)
    (values params
            (validate-positive-real-parameter caller :step-size step-size))))

(defun make-empty-diagnostics (n-warmup)
  "Create diagnostics for an inference call that returned no samples."
  (make-inference-diagnostics
   :accept-rate 0.0d0
   :n-divergences 0
   :final-step-size 0.0d0
   :n-samples 0
   :n-warmup n-warmup
   :elapsed-seconds 0.0d0))

(defun make-final-diagnostics (&key accept-rate n-divergences final-step-size
                                 n-samples n-warmup start-time)
  "Create the final diagnostics struct for an inference run."
  (make-inference-diagnostics
   :accept-rate (coerce accept-rate 'double-float)
   :n-divergences n-divergences
   :final-step-size (coerce final-step-size 'double-float)
   :n-samples n-samples
   :n-warmup n-warmup
   :elapsed-seconds (/ (float (- (get-internal-real-time) start-time) 0.0d0)
                       internal-time-units-per-second)))

(defun warn-high-divergence (n-divergences n-samples)
  "Warn when the divergence rate exceeds the configured threshold."
  (when (and (> n-divergences 0)
             (> (/ (float n-divergences 0.0d0)
                   (float (max 1 n-samples) 0.0d0))
                +high-divergence-threshold+))
    (restart-case
        (warn 'high-divergence-warning
              :n-divergences n-divergences
              :n-samples n-samples)
      (continue-with-warnings ()
        :report "Continue and return results despite high divergences"
        nil))))

(defun log-random-unit ()
  "Return log(U) for U sampled uniformly from (0, 1]."
  (log (max double-float-epsilon (random 1.0d0))))

(defun safe-gradient (log-pdf-fn q)
  "Call ad:gradient with traps masked.
Returns (values val grad) or (values nil nil) if evaluation is not finite."
  (let ((non-finite-p nil))
    (multiple-value-bind (val grad)
        (handler-case
            (with-float-traps-masked
              (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
                (let ((val-d (coerce val 'double-float)))
                  (if (and (finite-double-p val-d)
                           (every-finite-p grad))
                      (values val-d grad)
                      (progn
                        (setf non-finite-p t)
                        (values nil nil))))))
          (arithmetic-error () (values nil nil))
          (error (c)
            (warn-log-pdf-error-once "safe-gradient" c)
            (values nil nil)))
      (when non-finite-p
        (signal 'non-finite-gradient-error
                :params q
                :message "non-finite log-pdf value or gradient"))
      (values val grad))))

(defun resolve-initial-state (log-pdf-fn initial-params &key include-gradient)
  "Resolve a valid starting point using the shared restart protocol.
Returns (values params log-pdf grad) on success, or NIL values if the caller
chose the RETURN-EMPTY-SAMPLES restart."
  (let ((current-q (copy-list initial-params)))
    (loop
      (multiple-value-bind (val grad)
          (safe-gradient log-pdf-fn current-q)
        (when val
          (return (values current-q
                          val
                          (and include-gradient grad))))
        (multiple-value-bind (action new-params)
            (restart-case
                (error 'invalid-initial-params-error
                       :params (copy-list current-q)
                       :message "LOG-PDF-FN returned non-finite value at INITIAL-PARAMS")
              (use-fallback-params (new-params)
                :report "Supply new initial params and retry"
                :interactive (lambda ()
                               (format *query-io*
                                       "New initial params (a list of numbers): ")
                               (list (read *query-io*)))
                (values :retry new-params))
              (return-empty-samples ()
                :report "Return empty sample list"
                (values :empty nil)))
          (ecase action
            (:retry
             (setf current-q (validate-initial-params "inference" new-params)))
            (:empty
             (return (values nil nil nil)))))))))

(defun compute-kinetic-energy (momentum)
  "Kinetic energy: 0.5 * sum(p_i^2)."
  (* 0.5d0 (loop for p double-float in momentum sum (* p p))))
