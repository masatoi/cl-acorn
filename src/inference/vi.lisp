(in-package #:cl-acorn.inference)

(defvar +log-sigma-max+ 10.0d0
  "Maximum log-sigma to prevent exp overflow in VI reparameterization.")

(defvar +log-sigma-min+ -10.0d0
  "Minimum log-sigma to prevent underflow in VI reparameterization.")

(defvar +max-grad-magnitude+ 1d100
  "Maximum gradient magnitude to prevent g*g overflow in Adam's second moment.")

(defun clamp-log-sigmas (var-params n-params)
  "Clamp the log-sigma portion of VAR-PARAMS to [+log-sigma-min+, +log-sigma-max+].
VAR-PARAMS is [mu_1..mu_n, log_sigma_1..log_sigma_n]. O(n) traversal."
  (let ((result (copy-list var-params)))
    (loop for cell on (nthcdr n-params result)
          for v = (coerce (car cell) 'double-float)
          do (setf (car cell)
                   (if (finite-double-p v)
                       (max +log-sigma-min+ (min +log-sigma-max+ v))
                       0.0d0
                       )))
    result))

(defun clip-gradients (grads)
  "Clip gradient magnitudes to +max-grad-magnitude+ to prevent g*g overflow in Adam."
  (mapcar (lambda (g)
            (let ((gd (coerce g 'double-float)))
              (max (- +max-grad-magnitude+)
                   (min +max-grad-magnitude+ gd))))
          grads))

(defun sample-vi-epsilons (n-params n-elbo-samples)
  "Draw N-ELBO-SAMPLES standard-normal noise vectors for the VI objective."
  (loop repeat n-elbo-samples
        collect (loop repeat n-params
                      collect (dist:normal-sample))))

(defun negative-elbo (log-pdf-fn var-params n-params epsilons)
  "Estimate the negative ELBO for VAR-PARAMS using the sampled EPSILONS."
  (let* ((mus (subseq var-params 0 n-params))
         (log-sigmas (nthcdr n-params var-params))
         (sample-count (coerce (length epsilons) 'double-float))
         (zero-like (ad:* 0.0d0 (first var-params)))
         (total-log-p zero-like)
         (entropy (reduce #'ad:+ log-sigmas :initial-value zero-like)))
    (dolist (eps epsilons)
      (let ((z (mapcar (lambda (mu ls e)
                         (ad:+ mu (ad:* (ad:exp ls) e)))
                       mus log-sigmas eps)))
        (setf total-log-p
              (ad:+ total-log-p (funcall log-pdf-fn z)))))
    (ad:- (ad:+ (ad:/ total-log-p sample-count) entropy))))

(defun estimate-vi-gradient (log-pdf-fn var-params n-params epsilons)
  "Estimate the VI objective and its gradient for VAR-PARAMS."
  (handler-case
      (with-float-traps-masked
        (ad:gradient (lambda (vp)
                       (negative-elbo log-pdf-fn vp n-params epsilons))
                     var-params))
    (arithmetic-error () (values nil nil))
    (error (c)
      (warn-log-pdf-error-once "vi" c)
      (values nil nil))))

(defun vi-step (log-pdf-fn var-params n-params n-elbo-samples adam-state lr)
  "Run one VI optimization step.
Returns updated variational parameters, the ELBO for this step, and a success flag."
  (let ((epsilons (sample-vi-epsilons n-params n-elbo-samples)))
    (multiple-value-bind (neg-elbo grads)
        (estimate-vi-gradient log-pdf-fn var-params n-params epsilons)
      (if (and neg-elbo grads (every-finite-p grads))
          (let* ((clipped-grads (clip-gradients grads))
                 (updated-var-params
                   (clamp-log-sigmas
                    (opt:adam-step var-params clipped-grads adam-state :lr lr)
                    n-params)))
            (values updated-var-params
                    (- (coerce neg-elbo 'double-float))
                    t))
          (values var-params nil nil)))))

(defun extract-vi-results (var-params n-params)
  "Extract final mean and scale parameters from VAR-PARAMS."
  (let ((mu-list (mapcar (lambda (x) (coerce x 'double-float))
                         (subseq var-params 0 n-params)))
        (sigma-list (mapcar (lambda (x)
                              (exp (coerce x 'double-float)))
                            (subseq var-params n-params))))
    (when (some (lambda (m) (not (finite-double-p m))) mu-list)
      (warn "vi: Some mu values are non-finite. Results may be unreliable."))
    (values mu-list sigma-list)))

(defun vi (log-pdf-fn initial-params
           &key (n-iterations 1000) (n-elbo-samples 10) (lr 0.01d0))
  "Mean-field Automatic Differentiation Variational Inference (ADVI).
Approximates the posterior defined by LOG-PDF-FN with a factored Gaussian
q(z) = prod_i N(z_i | mu_i, sigma_i).

LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations.
INITIAL-PARAMS is a list of starting parameter values; its length determines
the number of parameters to infer.

Returns (values mu-list sigma-list elbo-history diagnostics) where:
  MU-LIST: posterior mean estimates
  SIGMA-LIST: posterior standard deviation estimates
  ELBO-HISTORY: list of ELBO values over iterations
  DIAGNOSTICS: an INFERENCE-DIAGNOSTICS struct with timing and summary stats."
  (let* ((initial-params (validate-initial-params "vi" initial-params))
         (n-params (length initial-params)))
    (validate-positive-integer-parameter "vi" :n-iterations n-iterations)
    (validate-positive-integer-parameter "vi" :n-elbo-samples n-elbo-samples)
    (setf lr (validate-positive-real-parameter "vi" :lr lr))
    ;; Variational parameters: [mu_1..mu_n, log_sigma_1..log_sigma_n]
    ;; packed as a flat list of length 2*n-params
    (let* ((*log-pdf-error-warned-p* nil)
           (start-time (get-internal-real-time))
           (n-var-params (* 2 n-params))
           (var-params (make-list n-var-params :initial-element 0.0d0))
           (adam-state (opt:make-adam-state n-var-params))
           (elbo-history nil))
      (with-float-traps-masked
        (dotimes (iter n-iterations)
          (declare (ignore iter))
          (multiple-value-bind (updated-var-params elbo successp)
              (vi-step log-pdf-fn var-params n-params n-elbo-samples adam-state lr)
            (setf var-params updated-var-params)
            (when successp
              (push elbo elbo-history)))))
      (when (null elbo-history)
        (warn "vi: All ~D iterations produced non-finite gradients. ~
               Results may be meaningless." n-iterations))
      (multiple-value-bind (mu-list sigma-list)
          (extract-vi-results var-params n-params)
        (values mu-list sigma-list (nreverse elbo-history)
                (make-final-diagnostics
                 :accept-rate 0.0d0
                 :n-divergences 0
                 :final-step-size lr
                 :n-samples n-iterations
                 :n-warmup 0
                 :start-time start-time))))))
