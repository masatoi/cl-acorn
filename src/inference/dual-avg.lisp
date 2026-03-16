(in-package #:cl-acorn.inference)

(defvar +log-epsilon-max+ 20.0d0
  "Maximum log step-size to prevent exp overflow (exp(20) ~ 4.8e8).")

(defvar +log-epsilon-min+ -20.0d0
  "Minimum log step-size to prevent underflow (exp(-20) ~ 2e-9).")

(defstruct (dual-avg-state (:constructor %make-dual-avg-state))
  "State for Nesterov dual averaging step-size adaptation.
Implements Algorithm 5 from Hoffman & Gelman (2014, 'The No-U-Turn Sampler')."
  (log-epsilon 0.0d0 :type double-float)
  (log-epsilon-bar 0.0d0 :type double-float)
  (h-bar 0.0d0 :type double-float)
  (mu 0.0d0 :type double-float)
  (gamma 0.05d0 :type double-float)
  (t0 10.0d0 :type double-float)
  (kappa 0.75d0 :type double-float)
  (target-accept 0.65d0 :type double-float)
  (step-count 0 :type fixnum))

(defun make-dual-avg-state (initial-step-size &key (target-accept 0.65d0))
  "Create a dual averaging state for step-size adaptation.
INITIAL-STEP-SIZE is the starting step-size (epsilon).
TARGET-ACCEPT is the desired average acceptance probability (default 0.65)."
  (assert (> initial-step-size 0.0d0) nil
          "make-dual-avg-state: INITIAL-STEP-SIZE must be positive")
  (assert (and (> target-accept 0.0d0) (< target-accept 1.0d0)) nil
          "make-dual-avg-state: TARGET-ACCEPT must be in (0, 1)")
  (let ((log-eps (log (coerce initial-step-size 'double-float))))
    (%make-dual-avg-state
     :log-epsilon log-eps
     :log-epsilon-bar 0.0d0
     :h-bar 0.0d0
     :mu (log (* 10.0d0 (coerce initial-step-size 'double-float)))
     :target-accept (coerce target-accept 'double-float))))

(defun dual-avg-update (state accept-prob)
  "Update the dual averaging state with a new acceptance probability.
ACCEPT-PROB is clamped to [0, 1] and NaN is treated as 0.
Returns the new step-size to use for the next iteration."
  (let* ((m (1+ (dual-avg-state-step-count state)))
         (target (dual-avg-state-target-accept state))
         (gamma (dual-avg-state-gamma state))
         (t0 (dual-avg-state-t0 state))
         (kappa (dual-avg-state-kappa state))
         (mu (dual-avg-state-mu state))
         ;; Clamp and sanitize accept-prob: NaN/Inf -> 0
         (accept-prob-d (coerce accept-prob 'double-float))
         (accept-prob (if (finite-double-p accept-prob-d)
                          (max 0.0d0 (min 1.0d0 accept-prob-d))
                          0.0d0))
         ;; Update H-bar: running mean of (target - accept-prob)
         (new-h-bar (+ (* (- 1.0d0 (/ 1.0d0 (+ m t0)))
                           (dual-avg-state-h-bar state))
                        (* (/ 1.0d0 (+ m t0))
                           (- target accept-prob))))
         ;; Update log-epsilon with clamping
         (new-log-eps (- mu (* (/ (sqrt (coerce m 'double-float)) gamma)
                               new-h-bar)))
         (new-log-eps (max +log-epsilon-min+ (min +log-epsilon-max+ new-log-eps)))
         ;; Update log-epsilon-bar (smoothed) with clamping
         (eta (expt (coerce m 'double-float) (- kappa)))
         (new-log-eps-bar (+ (* eta new-log-eps)
                             (* (- 1.0d0 eta)
                                (dual-avg-state-log-epsilon-bar state))))
         (new-log-eps-bar (max +log-epsilon-min+
                               (min +log-epsilon-max+ new-log-eps-bar))))
    (setf (dual-avg-state-step-count state) m
          (dual-avg-state-h-bar state) new-h-bar
          (dual-avg-state-log-epsilon state) new-log-eps
          (dual-avg-state-log-epsilon-bar state) new-log-eps-bar)
    (exp new-log-eps)))

(defun dual-avg-final-step-size (state)
  "Return the final smoothed step-size after adaptation is complete."
  (exp (dual-avg-state-log-epsilon-bar state)))
