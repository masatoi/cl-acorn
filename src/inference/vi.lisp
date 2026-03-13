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
                   (if (or (sb-ext:float-nan-p v)
                           (sb-ext:float-infinity-p v))
                       0.0d0
                       (max +log-sigma-min+ (min +log-sigma-max+ v)))))
    result))

(defun clip-gradients (grads)
  "Clip gradient magnitudes to +max-grad-magnitude+ to prevent g*g overflow in Adam."
  (mapcar (lambda (g)
            (let ((gd (coerce g 'double-float)))
              (max (- +max-grad-magnitude+)
                   (min +max-grad-magnitude+ gd))))
          grads))

(defun vi (log-pdf-fn n-params
           &key (n-iterations 1000) (n-elbo-samples 10) (lr 0.01d0))
  "Mean-field Automatic Differentiation Variational Inference (ADVI).
Approximates the posterior defined by LOG-PDF-FN with a factored Gaussian
q(z) = prod_i N(z_i | mu_i, sigma_i).

LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations.
N-PARAMS is the number of parameters to infer.

Returns (values mu-list sigma-list elbo-history diagnostics) where:
  MU-LIST: posterior mean estimates
  SIGMA-LIST: posterior standard deviation estimates
  ELBO-HISTORY: list of ELBO values over iterations
  DIAGNOSTICS: an INFERENCE-DIAGNOSTICS struct with timing and summary stats."
  (unless (and (integerp n-params) (plusp n-params))
    (error 'invalid-parameter-error
           :parameter :n-params :value n-params
           :message "vi: N-PARAMS must be a positive integer"))
  (unless (and (integerp n-iterations) (plusp n-iterations))
    (error 'invalid-parameter-error
           :parameter :n-iterations :value n-iterations
           :message "vi: N-ITERATIONS must be a positive integer"))
  (unless (and (integerp n-elbo-samples) (plusp n-elbo-samples))
    (error 'invalid-parameter-error
           :parameter :n-elbo-samples :value n-elbo-samples
           :message "vi: N-ELBO-SAMPLES must be a positive integer"))
  (unless (> lr 0.0d0)
    (error 'invalid-parameter-error
           :parameter :lr :value lr
           :message "vi: LR must be a positive number"))
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
      ;; Draw epsilon samples OUTSIDE gradient computation (plain numbers)
      (let ((epsilons (loop repeat n-elbo-samples
                            collect (loop repeat n-params
                                         collect (dist:normal-sample)))))
        ;; Compute ELBO gradient w.r.t. variational parameters
        (multiple-value-bind (neg-elbo grads)
            (handler-case
                (with-float-traps-masked
                  (ad:gradient
                   (lambda (vp)
                     (let* ((mus (subseq vp 0 n-params))
                            (log-sigmas (nthcdr n-params vp))
                            (total-log-p (ad:* 0.0d0 (first vp)))
                            (total-entropy (ad:* 0.0d0 (first vp))))
                       ;; Monte Carlo estimate of E_q[log p(z)]
                       (dolist (eps epsilons)
                         (let ((z (mapcar (lambda (mu ls e)
                                            (ad:+ mu (ad:* (ad:exp ls) e)))
                                          mus log-sigmas eps)))
                           (setf total-log-p
                                 (ad:+ total-log-p (funcall log-pdf-fn z)))))
                       (let ((mean-log-p (ad:/ total-log-p
                                               (coerce n-elbo-samples
                                                       'double-float))))
                         ;; Entropy of factored Gaussian: sum(log_sigma_i)
                         ;; Constant n/2*log(2*pi*e) does not affect gradients
                         (dolist (ls log-sigmas)
                           (setf total-entropy (ad:+ total-entropy ls)))
                         ;; Negate ELBO for minimization
                         (ad:- (ad:+ mean-log-p total-entropy)))))
                   var-params))
              (arithmetic-error () (values nil nil))
              (error (c)
                (unless *log-pdf-error-warned-p*
                  (warn "vi: caught ~A in log-pdf-fn: ~A~%~
                         Further non-arithmetic errors will be suppressed."
                        (type-of c) c)
                  (setf *log-pdf-error-warned-p* t))
                (values nil nil)))
          ;; Skip iteration if gradient computation failed
          (when (and neg-elbo grads
                     (every (lambda (g)
                              (let ((gd (coerce g 'double-float)))
                                (and (not (sb-ext:float-nan-p gd))
                                     (not (sb-ext:float-infinity-p gd)))))
                            grads))
            ;; Record ELBO (negate back since we computed -ELBO)
            (push (- (coerce neg-elbo 'double-float)) elbo-history)
            ;; Clip gradients to prevent g*g overflow in Adam's v computation
            (let ((clipped-grads (clip-gradients grads)))
              ;; Adam update
              (setf var-params (opt:adam-step var-params clipped-grads adam-state
                                             :lr (coerce lr 'double-float))))
            ;; Clamp log-sigma to prevent overflow on next iteration
            (setf var-params (clamp-log-sigmas var-params n-params)))))))
    ;; Warn if optimization never succeeded
    (when (null elbo-history)
      (warn "vi: All ~D iterations produced non-finite gradients. ~
             Results may be meaningless." n-iterations))
    ;; Extract final mu and sigma, with finiteness check on mu
    (let ((mu-list (mapcar (lambda (x) (coerce x 'double-float))
                           (subseq var-params 0 n-params)))
          (sigma-list (mapcar (lambda (x)
                                (exp (coerce x 'double-float)))
                              (subseq var-params n-params))))
      (when (some (lambda (m) (or (sb-ext:float-nan-p m)
                                   (sb-ext:float-infinity-p m)))
                  mu-list)
        (warn "vi: Some mu values are non-finite. Results may be unreliable."))
      (values mu-list sigma-list (nreverse elbo-history)
              (make-inference-diagnostics
               :accept-rate 0.0d0       ; not applicable for VI
               :n-divergences 0
               :final-step-size (coerce lr 'double-float)
               :n-samples n-iterations
               :n-warmup 0
               :elapsed-seconds (/ (float (- (get-internal-real-time) start-time)
                                          0.0d0)
                                   internal-time-units-per-second))))))
