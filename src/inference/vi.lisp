(in-package #:cl-acorn.inference)

(defun vi (log-pdf-fn n-params
           &key (n-iterations 1000) (n-elbo-samples 10) (lr 0.01d0))
  "Mean-field Automatic Differentiation Variational Inference (ADVI).
Approximates the posterior defined by LOG-PDF-FN with a factored Gaussian
q(z) = prod_i N(z_i | mu_i, sigma_i).

LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations.
N-PARAMS is the number of parameters to infer.

Returns (values mu-list sigma-list elbo-history) where:
  MU-LIST: posterior mean estimates
  SIGMA-LIST: posterior standard deviation estimates
  ELBO-HISTORY: list of ELBO values over iterations."
  (assert (and (integerp n-params) (plusp n-params)) nil
          "vi: N-PARAMS must be a positive integer")
  (assert (and (integerp n-iterations) (plusp n-iterations)) nil
          "vi: N-ITERATIONS must be a positive integer")
  (assert (and (integerp n-elbo-samples) (plusp n-elbo-samples)) nil
          "vi: N-ELBO-SAMPLES must be a positive integer")
  (assert (> lr 0.0d0) nil "vi: LR must be a positive number")
  ;; Variational parameters: [mu_1..mu_n, log_sigma_1..log_sigma_n]
  ;; packed as a flat list of length 2*n-params
  (let* ((n-var-params (* 2 n-params))
         (var-params (make-list n-var-params :initial-element 0.0d0))
         (adam-state (opt:make-adam-state n-var-params))
         (elbo-history nil))
    (dotimes (iter n-iterations)
      ;; Draw epsilon samples OUTSIDE gradient computation (plain numbers)
      (let ((epsilons (loop repeat n-elbo-samples
                            collect (loop repeat n-params
                                         collect (dist:normal-sample)))))
        ;; Compute ELBO gradient w.r.t. variational parameters
        (multiple-value-bind (neg-elbo grads)
            (ad:gradient
             (lambda (vp)
               ;; vp is a list of tape-nodes: [mu_1..mu_n, log_sigma_1..log_sigma_n]
               (let* ((mus (subseq vp 0 n-params))
                      (log-sigmas (subseq vp n-params))
                      (total-log-p (ad:* 0.0d0 (first vp))) ; zero as tape-node
                      (total-entropy (ad:* 0.0d0 (first vp))))
                 ;; Monte Carlo estimate of E_q[log p(z)]
                 (dolist (eps epsilons)
                   ;; Reparameterization: z = mu + exp(log_sigma) * epsilon
                   (let ((z (mapcar (lambda (mu ls e)
                                      (ad:+ mu (ad:* (ad:exp ls) e)))
                                    mus log-sigmas eps)))
                     (setf total-log-p (ad:+ total-log-p (funcall log-pdf-fn z)))))
                 (let ((mean-log-p (ad:/ total-log-p
                                         (coerce n-elbo-samples 'double-float))))
                   ;; Entropy of factored Gaussian: sum(log_sigma_i) + n/2*log(2*pi*e)
                   (dolist (ls log-sigmas)
                     (setf total-entropy (ad:+ total-entropy ls)))
                   ;; Negate ELBO for minimization: -(E[log p] + entropy)
                   (ad:- (ad:+ mean-log-p total-entropy)))))
             var-params)
          ;; Record ELBO (negate back since we computed -ELBO)
          (push (- (coerce neg-elbo 'double-float)) elbo-history)
          ;; Adam update (Adam minimizes, and we computed -ELBO gradient)
          (setf var-params (opt:adam-step var-params grads adam-state
                                         :lr (coerce lr 'double-float))))))
    ;; Extract final mu and sigma
    (let ((mu-list (mapcar (lambda (x) (coerce x 'double-float))
                           (subseq var-params 0 n-params)))
          (sigma-list (mapcar (lambda (x) (exp (coerce x 'double-float)))
                              (subseq var-params n-params))))
      (values mu-list sigma-list (nreverse elbo-history)))))
