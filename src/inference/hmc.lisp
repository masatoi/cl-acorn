(in-package #:cl-acorn.inference)

(declaim (inline finite-double-p))
(defun finite-double-p (x)
  "Return T if X is a finite double-float (not NaN, not infinity)."
  (and (typep x 'double-float)
       (not (sb-ext:float-nan-p x))
       (not (sb-ext:float-infinity-p x))))

(defun every-finite-p (list)
  "Return T if every element in LIST is a finite double-float."
  (every #'finite-double-p list))

(defmacro with-float-traps-masked (&body body)
  "Execute BODY with floating-point traps masked to allow Inf/NaN creation."
  `(sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
     ,@body))

(defvar *log-pdf-error-warned-p* nil
  "When non-nil, suppresses repeated warnings about non-arithmetic errors in log-pdf-fn.
Bound to NIL at the start of each top-level inference call (hmc, nuts, vi).")

(defun safe-gradient (log-pdf-fn q)
  "Call ad:gradient with float traps masked. Returns (values val grad) or
(values nil nil) if any error occurs (arithmetic or user function errors).
Warns once per inference call for non-arithmetic errors."
  (handler-case
      (with-float-traps-masked
        (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
          (let ((val-d (coerce val 'double-float)))
            (if (and (finite-double-p val-d)
                     (every-finite-p grad))
                (values val-d grad)
                (values nil nil)))))
    (arithmetic-error () (values nil nil))
    (error (c)
      (unless *log-pdf-error-warned-p*
        (warn "safe-gradient: caught ~A in log-pdf-fn: ~A~%~
               Further non-arithmetic errors will be suppressed for this call."
              (type-of c) c)
        (setf *log-pdf-error-warned-p* t))
      (values nil nil))))

(defun compute-kinetic-energy (momentum)
  "Kinetic energy: 0.5 * sum(p_i^2)."
  (* 0.5d0 (loop for p double-float in momentum sum (* p p))))

(defun leapfrog-step (log-pdf-fn q p step-size &optional grad)
  "Single leapfrog integration step.
Takes position Q, momentum P, STEP-SIZE, and optional precomputed gradient GRAD.
Returns (values q-new p-new log-pdf-new grad-new).
If overflow, NaN, or any error is detected, returns (values nil nil nil nil)."
  (handler-case
      (with-float-traps-masked
        (let ((g (or grad
                     (nth-value 1 (ad:gradient log-pdf-fn q)))))
          (unless (every-finite-p g)
            (return-from leapfrog-step (values nil nil nil nil)))
          (let ((p-half (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi))) p g)))
            (unless (every-finite-p p-half)
              (return-from leapfrog-step (values nil nil nil nil)))
            (let ((q-new (mapcar (lambda (qi pv) (+ qi (* step-size pv))) q p-half)))
              (unless (every-finite-p q-new)
                (return-from leapfrog-step (values nil nil nil nil)))
              (multiple-value-bind (new-val new-grad) (ad:gradient log-pdf-fn q-new)
                (let ((new-val-d (coerce new-val 'double-float)))
                  (unless (and (finite-double-p new-val-d)
                               (every-finite-p new-grad))
                    (return-from leapfrog-step (values nil nil nil nil)))
                  (let ((p-new (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi)))
                                       p-half new-grad)))
                    (unless (every-finite-p p-new)
                      (return-from leapfrog-step (values nil nil nil nil)))
                    (values q-new p-new new-val-d new-grad))))))))
    (arithmetic-error () (values nil nil nil nil))
    (error (c)
      (unless *log-pdf-error-warned-p*
        (warn "leapfrog-step: caught ~A in log-pdf-fn: ~A~%~
               Further non-arithmetic errors will be suppressed for this call."
              (type-of c) c)
        (setf *log-pdf-error-warned-p* t))
      (values nil nil nil nil))))

(defun leapfrog (log-pdf-fn q p step-size n-steps)
  "Leapfrog integrator for Hamiltonian dynamics.
LOG-PDF-FN accepts a parameter list and returns a scalar.
Q and P are lists of position and momentum values.
Returns (values q-new p-new log-pdf-new) or (values nil nil nil) on divergence."
  (handler-case
      (with-float-traps-masked
        (let ((q (copy-list q))
              (p (copy-list p)))
          ;; Half step for momentum
          (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
            (declare (ignore val))
            (unless (every-finite-p grad)
              (return-from leapfrog (values nil nil nil)))
            (setf p (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi))) p grad)))
          (unless (every-finite-p p)
            (return-from leapfrog (values nil nil nil)))
          ;; Full steps
          (dotimes (unused (1- n-steps))
            (declare (ignorable unused))
            (setf q (mapcar (lambda (qi pv) (+ qi (* step-size pv))) q p))
            (unless (every-finite-p q)
              (return-from leapfrog (values nil nil nil)))
            (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
              (declare (ignore val))
              (setf p (mapcar (lambda (pv gi) (+ pv (* step-size gi))) p grad)))
            (unless (every-finite-p p)
              (return-from leapfrog (values nil nil nil))))
          ;; Final full step for position
          (setf q (mapcar (lambda (qi pv) (+ qi (* step-size pv))) q p))
          (unless (every-finite-p q)
            (return-from leapfrog (values nil nil nil)))
          ;; Half step for momentum
          (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
            (let ((val-d (coerce val 'double-float)))
              (setf p (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi))) p grad))
              (unless (and (finite-double-p val-d) (every-finite-p p))
                (return-from leapfrog (values nil nil nil)))
              (values q p val-d)))))
    (arithmetic-error () (values nil nil nil))
    (error (c)
      (unless *log-pdf-error-warned-p*
        (warn "leapfrog: caught ~A in log-pdf-fn: ~A~%~
               Further non-arithmetic errors will be suppressed for this call."
              (type-of c) c)
        (setf *log-pdf-error-warned-p* t))
      (values nil nil nil))))

(defun hmc (log-pdf-fn initial-params
            &key (n-samples 1000) (n-warmup 500)
                 (step-size 0.01d0) (n-leapfrog 10)
                 (adapt-step-size nil))
  "Hamiltonian Monte Carlo sampler.
LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations (for gradient computation via ad:gradient).
INITIAL-PARAMS is a list of starting parameter values.
When ADAPT-STEP-SIZE is true, uses dual averaging to adapt step-size during warmup.
Returns (values samples accept-rate) where samples is a list of parameter lists
and accept-rate is the fraction of accepted proposals after warmup."
  (assert (and (listp initial-params) (consp initial-params)) nil
          "hmc: INITIAL-PARAMS must be a non-empty list")
  (assert (and (integerp n-samples) (plusp n-samples)) nil
          "hmc: N-SAMPLES must be a positive integer")
  (assert (and (integerp n-warmup) (not (minusp n-warmup))) nil
          "hmc: N-WARMUP must be a non-negative integer")
  (assert (> step-size 0.0d0) nil "hmc: STEP-SIZE must be a positive number")
  (assert (and (integerp n-leapfrog) (plusp n-leapfrog)) nil
          "hmc: N-LEAPFROG must be a positive integer")
  (let* ((*log-pdf-error-warned-p* nil)
         (current-q (mapcar (lambda (x) (coerce x 'double-float)) initial-params))
         (n-dim (length current-q))
         (samples nil)
         (n-accepted 0)
         (total-iterations (+ n-samples n-warmup))
         (step-size (coerce step-size 'double-float))
         (da-state (when (and adapt-step-size (plusp n-warmup))
                     (make-dual-avg-state step-size
                                          :target-accept 0.65d0)))
         ;; Cache current log-pdf to avoid redundant gradient evaluations
         (current-log-pdf (multiple-value-bind (val grad)
                              (safe-gradient log-pdf-fn current-q)
                            (declare (ignore grad))
                            (assert val nil
                                    "hmc: LOG-PDF-FN returned non-finite value at ~
                                     INITIAL-PARAMS. Ensure initial parameters are ~
                                     in the support of the distribution.")
                            val)))
    (with-float-traps-masked
     (dotimes (iter total-iterations)
      (let* ((current-p (loop repeat n-dim collect (dist:normal-sample)))
             (current-h (- (compute-kinetic-energy current-p) current-log-pdf)))
        (multiple-value-bind (proposed-q proposed-p proposed-log-pdf)
            (leapfrog log-pdf-fn current-q current-p step-size n-leapfrog)
          (cond
            ;; Diverged: reject proposal, adapt with accept-prob=0
            ((null proposed-q)
             (when (and da-state (< iter n-warmup))
               (setf step-size (dual-avg-update da-state 0.0d0))))
            ;; Current state non-finite: always accept finite proposals
            ((not (finite-double-p current-h))
             (setf current-q proposed-q
                   current-log-pdf proposed-log-pdf)
             (when (and da-state (< iter n-warmup))
               (setf step-size (dual-avg-update da-state 1.0d0)))
             (when (>= iter n-warmup)
               (incf n-accepted)))
            ;; Normal case: compute MH acceptance
            (t
             (let* ((proposed-h (- (compute-kinetic-energy proposed-p)
                                   proposed-log-pdf))
                    (log-accept-prob (- current-h proposed-h))
                    (log-accept-prob (if (finite-double-p log-accept-prob)
                                         log-accept-prob
                                         most-negative-double-float))
                    (accept-prob (min 1.0d0 (exp (min 0.0d0 log-accept-prob)))))
               (when (and da-state (< iter n-warmup))
                 (setf step-size (dual-avg-update da-state accept-prob)))
               (when (or (>= log-accept-prob 0.0d0)
                         (< (log (max double-float-epsilon (random 1.0d0)))
                            log-accept-prob))
                 (setf current-q proposed-q
                       current-log-pdf proposed-log-pdf)
                 (when (>= iter n-warmup)
                   (incf n-accepted))))))))
      ;; Finalize step-size at end of warmup
      (when (and da-state (= iter (1- n-warmup)))
        (setf step-size (dual-avg-final-step-size da-state)))
      ;; Collect sample after warmup
      (when (>= iter n-warmup)
        (push (copy-list current-q) samples))))
    (values (nreverse samples)
            (/ (coerce n-accepted 'double-float)
               (coerce n-samples 'double-float)))))
