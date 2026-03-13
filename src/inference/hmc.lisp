(in-package #:cl-acorn.inference)

(defun compute-kinetic-energy (momentum)
  "Kinetic energy: 0.5 * sum(p_i^2)."
  (* 0.5d0 (reduce #'+ momentum :key (lambda (p) (* p p)))))

(defun leapfrog (log-pdf-fn q p step-size n-steps)
  "Leapfrog integrator for Hamiltonian dynamics.
LOG-PDF-FN accepts a parameter list and returns a scalar.
Q and P are lists of position and momentum values.
Returns (values q-new p-new)."
  (let ((q (copy-list q))
        (p (copy-list p)))
    ;; Half step for momentum
    (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
      (declare (ignore val))
      (setf p (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi))) p grad)))
    ;; Full steps
    (dotimes (unused (1- n-steps))
      (declare (ignorable unused))
      ;; Full step for position
      (setf q (mapcar (lambda (qi pv) (+ qi (* step-size pv))) q p))
      ;; Full step for momentum
      (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
        (declare (ignore val))
        (setf p (mapcar (lambda (pv gi) (+ pv (* step-size gi))) p grad))))
    ;; Final full step for position
    (setf q (mapcar (lambda (qi pv) (+ qi (* step-size pv))) q p))
    ;; Half step for momentum
    (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
      (declare (ignore val))
      (setf p (mapcar (lambda (pv gi) (+ pv (* 0.5d0 step-size gi))) p grad)))
    (values q p)))

(defun hmc (log-pdf-fn initial-params
            &key (n-samples 1000) (n-warmup 500)
                 (step-size 0.01d0) (n-leapfrog 10))
  "Hamiltonian Monte Carlo sampler.
LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations (for gradient computation via ad:gradient).
INITIAL-PARAMS is a list of starting parameter values.
Returns (values samples accept-rate) where samples is a list of parameter lists
and accept-rate is the fraction of accepted proposals after warmup."
  (assert (and (listp initial-params) (plusp (length initial-params))) nil
          "hmc: INITIAL-PARAMS must be a non-empty list")
  (assert (and (integerp n-samples) (plusp n-samples)) nil
          "hmc: N-SAMPLES must be a positive integer")
  (assert (and (integerp n-warmup) (not (minusp n-warmup))) nil
          "hmc: N-WARMUP must be a non-negative integer")
  (assert (> step-size 0.0d0) nil "hmc: STEP-SIZE must be a positive number")
  (assert (and (integerp n-leapfrog) (plusp n-leapfrog)) nil
          "hmc: N-LEAPFROG must be a positive integer")
  (let ((current-q (mapcar (lambda (x) (coerce x 'double-float)) initial-params))
        (n-dim (length initial-params))
        (samples nil)
        (n-accepted 0)
        (total-iterations (+ n-samples n-warmup)))
    (dotimes (iter total-iterations)
      ;; Sample random momentum from N(0, 1)
      (let ((current-p (loop repeat n-dim collect (dist:normal-sample))))
        ;; Current Hamiltonian: H = -log-pdf(q) + 0.5*sum(p^2)
        (let* ((current-log-pdf
                 (coerce (nth-value 0 (ad:gradient log-pdf-fn current-q))
                         'double-float))
               (current-h (- (compute-kinetic-energy current-p) current-log-pdf)))
          ;; Leapfrog integration
          (multiple-value-bind (proposed-q proposed-p)
              (leapfrog log-pdf-fn current-q current-p step-size n-leapfrog)
            ;; Proposed Hamiltonian
            (let* ((proposed-log-pdf
                     (coerce (nth-value 0 (ad:gradient log-pdf-fn proposed-q))
                             'double-float))
                   (proposed-h (- (compute-kinetic-energy proposed-p)
                                  proposed-log-pdf))
                   (log-accept-prob (- current-h proposed-h)))
              ;; Metropolis accept/reject
              (when (or (>= log-accept-prob 0.0d0)
                        (< (log (max double-float-epsilon (random 1.0d0)))
                           log-accept-prob))
                (setf current-q proposed-q)
                (when (>= iter n-warmup)
                  (incf n-accepted)))))))
      ;; Collect sample after warmup
      (when (>= iter n-warmup)
        (push (copy-list current-q) samples)))
    (values (nreverse samples)
            (/ (coerce n-accepted 'double-float)
               (coerce n-samples 'double-float)))))
