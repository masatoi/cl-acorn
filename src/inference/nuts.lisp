(in-package #:cl-acorn.inference)

(defvar +max-delta-energy+ 1000.0d0
  "Maximum energy difference before declaring divergence.")

(defvar +high-divergence-threshold+ 0.10d0
  "Warn when post-warmup divergence rate exceeds this fraction (default 10%).")

(defun log-sum-exp (a b)
  "Numerically stable log(exp(a) + exp(b))."
  (let ((m (max a b)))
    (if (= m most-negative-double-float)
        most-negative-double-float
        (+ m (log (+ (exp (- a m)) (exp (- b m))))))))

(defun no-u-turn-p (q-minus p-minus q-plus p-plus)
  "Check the no-U-turn criterion.
Returns true if the trajectory should stop (U-turn detected)."
  (let ((dot-plus 0.0d0)
        (dot-minus 0.0d0))
    (loop for qm in q-minus
          for qp in q-plus
          for pp in p-plus
          for pm in p-minus
          for dq double-float = (- (the double-float qp) (the double-float qm))
          do (incf dot-plus (* dq (the double-float pp)))
             (incf dot-minus (* dq (the double-float pm))))
    (or (<= dot-plus 0.0d0) (<= dot-minus 0.0d0))))

(defstruct (tree-state (:constructor %make-tree-state))
  "State returned from build-tree: trajectory endpoints, proposal, and statistics."
  (q-minus nil :type list)
  (p-minus nil :type list)
  (grad-minus nil :type list)
  (q-plus nil :type list)
  (p-plus nil :type list)
  (grad-plus nil :type list)
  (q-proposal nil :type list)
  (log-pdf-proposal 0.0d0 :type double-float)
  (grad-proposal nil :type list)
  (log-sum-weight most-negative-double-float :type double-float)
  (n-valid 0 :type fixnum)
  (diverging-p nil :type boolean)
  (turning-p nil :type boolean)
  (alpha-sum 0.0d0 :type double-float)
  (n-alpha 0 :type fixnum))

(defun build-tree (log-pdf-fn q p grad log-pdf-q step-size direction depth
                   initial-energy)
  "Recursive balanced binary tree of leapfrog steps for NUTS.
Base case (depth=0): single leapfrog-step, check divergence.
Recursive: build inner half, then outer half, combine with multinomial sampling.
Returns a tree-state."
  (if (= depth 0)
      ;; Base case: single leapfrog step
      (multiple-value-bind (q-new p-new log-pdf-new grad-new)
          (leapfrog-step log-pdf-fn q p (* direction step-size) grad)
        ;; Handle leapfrog overflow (nil return)
        (if (null q-new)
            (%make-tree-state
             :q-minus q :p-minus p :grad-minus grad
             :q-plus q :p-plus p :grad-plus grad
             :q-proposal q
             :log-pdf-proposal (coerce (or log-pdf-q 0.0d0) 'double-float)
             :grad-proposal grad
             :log-sum-weight most-negative-double-float
             :n-valid 0
             :diverging-p t
             :turning-p nil
             :alpha-sum 0.0d0
             :n-alpha 1)
            ;; Normal case
            (let* ((kinetic (compute-kinetic-energy p-new))
                   (new-energy (- kinetic log-pdf-new))
                   (delta-energy (- new-energy initial-energy))
                   (diverging (or (> delta-energy +max-delta-energy+)
                                  (not (finite-double-p delta-energy))))
                   (log-weight (if diverging
                                   most-negative-double-float
                                   (- initial-energy new-energy)))
                   (accept-stat (if diverging
                                    0.0d0
                                    (min 1.0d0
                                         (exp (min 0.0d0
                                                   (- initial-energy new-energy)))))))
              (%make-tree-state
               :q-minus q-new :p-minus p-new :grad-minus grad-new
               :q-plus q-new :p-plus p-new :grad-plus grad-new
               :q-proposal q-new
               :log-pdf-proposal log-pdf-new
               :grad-proposal grad-new
               :log-sum-weight log-weight
               :n-valid (if diverging 0 1)
               :diverging-p diverging
               :turning-p nil
               :alpha-sum accept-stat
               :n-alpha 1))))
      ;; Recursive case: build two halves
      (let ((inner (build-tree log-pdf-fn q p grad log-pdf-q
                               step-size direction (1- depth)
                               initial-energy)))
        (if (or (tree-state-diverging-p inner)
                (tree-state-turning-p inner))
            inner
            ;; Build outer half from the appropriate endpoint
            (let ((outer
                    (if (= direction -1)
                        (build-tree log-pdf-fn
                                    (tree-state-q-minus inner)
                                    (tree-state-p-minus inner)
                                    (tree-state-grad-minus inner)
                                    log-pdf-q
                                    step-size direction (1- depth)
                                    initial-energy)
                        (build-tree log-pdf-fn
                                    (tree-state-q-plus inner)
                                    (tree-state-p-plus inner)
                                    (tree-state-grad-plus inner)
                                    log-pdf-q
                                    step-size direction (1- depth)
                                    initial-energy))))
              (cond
                ;; Outer diverged: keep inner's proposal, mark as diverging
                ((tree-state-diverging-p outer)
                 (setf (tree-state-diverging-p inner) t
                       (tree-state-alpha-sum inner)
                       (+ (tree-state-alpha-sum inner)
                          (tree-state-alpha-sum outer))
                       (tree-state-n-alpha inner)
                       (+ (tree-state-n-alpha inner)
                          (tree-state-n-alpha outer)))
                 inner)
                ;; Outer valid (may or may not have turned):
                ;; Combine weights and do multinomial sampling BEFORE checking U-turn.
                ;; A turned subtree's states are valid trajectory points; the turning
                ;; flag only signals to stop further extension.
                (t
                 (let* ((combined-log-weight
                          (log-sum-exp (tree-state-log-sum-weight inner)
                                       (tree-state-log-sum-weight outer)))
                        ;; Multinomial sampling: accept outer proposal with
                        ;; probability proportional to its weight
                        (accept-outer-p
                          (and (> (tree-state-log-sum-weight outer)
                                  most-negative-double-float)
                               (< (log (max double-float-epsilon
                                            (random 1.0d0)))
                                  (- (tree-state-log-sum-weight outer)
                                     combined-log-weight)))))
                   ;; Update endpoints based on direction
                   (if (= direction -1)
                       (setf (tree-state-q-minus inner)
                             (tree-state-q-minus outer)
                             (tree-state-p-minus inner)
                             (tree-state-p-minus outer)
                             (tree-state-grad-minus inner)
                             (tree-state-grad-minus outer))
                       (setf (tree-state-q-plus inner)
                             (tree-state-q-plus outer)
                             (tree-state-p-plus inner)
                             (tree-state-p-plus outer)
                             (tree-state-grad-plus inner)
                             (tree-state-grad-plus outer)))
                   ;; Update proposal if outer wins
                   (when accept-outer-p
                     (setf (tree-state-q-proposal inner)
                           (tree-state-q-proposal outer)
                           (tree-state-log-pdf-proposal inner)
                           (tree-state-log-pdf-proposal outer)
                           (tree-state-grad-proposal inner)
                           (tree-state-grad-proposal outer)))
                   ;; Combine statistics and check U-turn on combined trajectory
                   (setf (tree-state-log-sum-weight inner) combined-log-weight
                         (tree-state-n-valid inner)
                         (+ (tree-state-n-valid inner)
                            (tree-state-n-valid outer))
                         (tree-state-alpha-sum inner)
                         (+ (tree-state-alpha-sum inner)
                            (tree-state-alpha-sum outer))
                         (tree-state-n-alpha inner)
                         (+ (tree-state-n-alpha inner)
                            (tree-state-n-alpha outer))
                         ;; Turning if outer turned OR combined trajectory U-turns
                         (tree-state-turning-p inner)
                         (or (tree-state-turning-p outer)
                             (no-u-turn-p
                              (tree-state-q-minus inner)
                              (tree-state-p-minus inner)
                              (tree-state-q-plus inner)
                              (tree-state-p-plus inner))))
                   inner))))))))

(defun nuts (log-pdf-fn initial-params
             &key (n-samples 1000) (n-warmup 500)
                  (step-size 0.01d0) (max-tree-depth 10)
                  (adapt-step-size t))
  "No-U-Turn Sampler (NUTS) with multinomial sampling.
LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations (for gradient computation via ad:gradient).
INITIAL-PARAMS is a list of starting parameter values.
MAX-TREE-DEPTH limits the binary tree depth (max 2^depth leapfrog steps).
When ADAPT-STEP-SIZE is true (default), uses dual averaging during warmup
with target acceptance rate 0.80.
Returns (values samples accept-rate diagnostics) where SAMPLES is a list of
parameter lists, ACCEPT-RATE is the mean acceptance probability, and DIAGNOSTICS
is an INFERENCE-DIAGNOSTICS struct with timing, divergence count, and step-size."
  (assert (and (listp initial-params) (consp initial-params)) nil
          "nuts: INITIAL-PARAMS must be a non-empty list")
  (assert (and (integerp n-samples) (plusp n-samples)) nil
          "nuts: N-SAMPLES must be a positive integer")
  (assert (and (integerp n-warmup) (not (minusp n-warmup))) nil
          "nuts: N-WARMUP must be a non-negative integer")
  (assert (> step-size 0.0d0) nil "nuts: STEP-SIZE must be a positive number")
  (assert (and (integerp max-tree-depth) (plusp max-tree-depth)
               (<= max-tree-depth 25)) nil
          "nuts: MAX-TREE-DEPTH must be a positive integer <= 25")
  (let* ((*log-pdf-error-warned-p* nil)
         (current-q (mapcar (lambda (x) (coerce x 'double-float)) initial-params))
         (n-dim (length current-q))
         (samples nil)
         (total-iterations (+ n-samples n-warmup))
         (step-size (coerce step-size 'double-float))
         (da-state (when (and adapt-step-size (plusp n-warmup))
                     (make-dual-avg-state step-size :target-accept 0.80d0)))
         (sum-accept 0.0d0)
         (n-accept-total 0)
         (n-divergences 0)
         (start-time (get-internal-real-time))
         (current-log-pdf nil)
         (current-grad nil))
    ;; Validate initial params; offer restarts on failure
    (block validate
      (loop
        (multiple-value-bind (val grad)
            (safe-gradient log-pdf-fn current-q)
          (when val
            (setf current-log-pdf val current-grad grad)
            (return-from validate))
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
              (setf current-q
                    (mapcar (lambda (x) (coerce x 'double-float)) new-params)))
            (return-empty-samples ()
              :report "Return empty sample list"
              (return-from nuts
                (values '() 0.0d0
                        (make-inference-diagnostics
                         :n-samples 0 :n-warmup n-warmup))))))))
    (with-float-traps-masked
      (dotimes (iter total-iterations)
          ;; Sample random momentum
          (let* ((current-p (loop repeat n-dim collect (dist:normal-sample)))
                 (initial-energy (- (compute-kinetic-energy current-p)
                                    current-log-pdf))
                 ;; Initialize trajectory endpoints from current state
                 (q-minus (copy-list current-q))
                 (p-minus (copy-list current-p))
                 (grad-minus (copy-list current-grad))
                 (q-plus (copy-list current-q))
                 (p-plus (copy-list current-p))
                 (grad-plus (copy-list current-grad))
                 ;; Running log-sum-weight (initial state has weight 1 => log-weight 0)
                 (running-log-sum-weight 0.0d0)
                 ;; Per-iteration acceptance statistics
                 (iter-alpha-sum 0.0d0)
                 (iter-n-alpha 0))
            ;; Build tree by doubling, extending trajectory from endpoints
            (let ((depth 0)
                  (keep-going t)
                  (iter-diverged-p nil))
              (loop while (and keep-going (< depth max-tree-depth))
                    do (let* ((direction (if (< (random 1.0d0) 0.5d0) -1 1))
                              ;; Build subtree from appropriate trajectory endpoint
                              (tree (if (= direction -1)
                                        (build-tree log-pdf-fn
                                                    q-minus p-minus grad-minus
                                                    current-log-pdf
                                                    step-size direction depth
                                                    initial-energy)
                                        (build-tree log-pdf-fn
                                                    q-plus p-plus grad-plus
                                                    current-log-pdf
                                                    step-size direction depth
                                                    initial-energy))))
                         ;; Track acceptance statistics
                         (incf iter-alpha-sum (tree-state-alpha-sum tree))
                         (incf iter-n-alpha (tree-state-n-alpha tree))
                         (cond
                           ;; Subtree diverged: stop growing
                           ((tree-state-diverging-p tree)
                            (setf keep-going nil
                                  iter-diverged-p t))
                           (t
                            ;; Update trajectory endpoints from subtree
                            (if (= direction -1)
                                (setf q-minus (tree-state-q-minus tree)
                                      p-minus (tree-state-p-minus tree)
                                      grad-minus (tree-state-grad-minus tree))
                                (setf q-plus (tree-state-q-plus tree)
                                      p-plus (tree-state-p-plus tree)
                                      grad-plus (tree-state-grad-plus tree)))
                            ;; Multinomial sampling across depth levels.
                            ;; Include turned subtrees: their states are valid.
                            (when (> (tree-state-log-sum-weight tree)
                                     most-negative-double-float)
                              (let ((combined (log-sum-exp
                                               running-log-sum-weight
                                               (tree-state-log-sum-weight tree))))
                                (when (< (log (max double-float-epsilon
                                                   (random 1.0d0)))
                                         (- (tree-state-log-sum-weight tree)
                                            combined))
                                  (setf current-q (tree-state-q-proposal tree)
                                        current-log-pdf
                                        (tree-state-log-pdf-proposal tree)
                                        current-grad
                                        (tree-state-grad-proposal tree)))
                                (setf running-log-sum-weight combined)))
                            ;; Check stopping: subtree turned or full trajectory U-turn
                            (when (or (tree-state-turning-p tree)
                                      (no-u-turn-p q-minus p-minus
                                                   q-plus p-plus))
                              (setf keep-going nil))))
                         (incf depth)))
              ;; Count divergences in sampling phase only
              (when (and iter-diverged-p (>= iter n-warmup))
                (incf n-divergences)))
            ;; Accumulate stats for adaptation
            (incf sum-accept iter-alpha-sum)
            (incf n-accept-total iter-n-alpha))
          ;; Step-size adaptation during warmup
          (when (and da-state (< iter n-warmup))
            (let ((iter-accept (if (> n-accept-total 0)
                                   (/ sum-accept
                                      (coerce n-accept-total 'double-float))
                                   0.0d0)))
              (setf step-size (dual-avg-update da-state iter-accept))
              (setf sum-accept 0.0d0
                    n-accept-total 0)))
          ;; Finalize step-size at end of warmup
          (when (and da-state (= iter (1- n-warmup)))
            (setf step-size (dual-avg-final-step-size da-state)))
          ;; Reset acceptance counters at warmup→sampling transition
          (when (and (plusp n-warmup) (= iter (1- n-warmup)))
            (setf sum-accept 0.0d0
                  n-accept-total 0))
          ;; Collect sample after warmup
          (when (>= iter n-warmup)
            (push (copy-list current-q) samples))))
    ;; Warn if divergence rate exceeds threshold
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
          nil)))
    ;; Compute average accept rate over sampling phase
    (let ((final-accept (if (> n-accept-total 0)
                            (/ sum-accept (coerce n-accept-total 'double-float))
                            0.0d0)))
      (values (nreverse samples)
              final-accept
              (make-inference-diagnostics
               :accept-rate final-accept
               :n-divergences n-divergences
               :final-step-size step-size
               :n-samples n-samples
               :n-warmup n-warmup
               :elapsed-seconds (/ (float (- (get-internal-real-time) start-time)
                                          0.0d0)
                                   internal-time-units-per-second))))))
