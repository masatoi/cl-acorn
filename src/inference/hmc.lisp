(in-package #:cl-acorn.inference)

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
      (warn-log-pdf-error-once "leapfrog-step" c)
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
      (warn-log-pdf-error-once "leapfrog" c)
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
Returns (values samples accept-rate diagnostics) where SAMPLES is a list of
parameter lists, ACCEPT-RATE is the fraction of accepted proposals after warmup,
and DIAGNOSTICS is an INFERENCE-DIAGNOSTICS struct with timing and summary stats."
  (validate-positive-integer-parameter "hmc" :n-leapfrog n-leapfrog)
  (multiple-value-bind (initial-params step-size)
      (validate-mcmc-common-args "hmc" initial-params n-samples n-warmup step-size)
    (let* ((*log-pdf-error-warned-p* nil)
           (start-time (get-internal-real-time))
           (samples nil)
           (n-accepted 0)
           (n-divergences 0)
           (total-iterations (+ n-samples n-warmup))
           (da-state (when (and adapt-step-size (plusp n-warmup))
                       (make-dual-avg-state step-size
                                            :target-accept 0.65d0)))
           (current-q nil)
           (current-log-pdf nil))
      (multiple-value-bind (resolved-q resolved-log-pdf ignored-grad)
          (resolve-initial-state log-pdf-fn initial-params)
        (declare (ignore ignored-grad))
        (unless resolved-q
          (return-from hmc
            (values '() 0.0d0 (make-empty-diagnostics n-warmup))))
        (setf current-q resolved-q
              current-log-pdf resolved-log-pdf))
      (let ((n-dim (length current-q)))
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
                 (setf step-size (dual-avg-update da-state 0.0d0)))
               (when (>= iter n-warmup)
                 (incf n-divergences)))
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
                           (< (log-random-unit)
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
      (warn-high-divergence n-divergences n-samples)
      (let ((accept-rate (/ (coerce n-accepted 'double-float)
                            (coerce n-samples 'double-float))))
        (values (nreverse samples)
                accept-rate
                (make-final-diagnostics
                 :accept-rate accept-rate
                 :n-divergences n-divergences
                 :final-step-size step-size
                 :n-samples n-samples
                 :n-warmup n-warmup
                 :start-time start-time)))))))
