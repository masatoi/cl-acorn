(in-package #:cl-acorn.inference)

(defvar +max-delta-energy+ 1000.0d0
  "Maximum energy difference before declaring divergence.")

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

(defun build-tree-leaf (log-pdf-fn q p grad log-pdf-q step-size direction initial-energy)
  "Build the depth-0 NUTS subtree rooted at Q/P."
  (multiple-value-bind (q-new p-new log-pdf-new grad-new)
      (leapfrog-step log-pdf-fn q p (* direction step-size) grad)
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
           :n-alpha 1)))))

(defun merge-tree-states (inner outer direction)
  "Merge INNER and OUTER tree states after recursive expansion."
  (cond
    ((tree-state-diverging-p outer)
     (setf (tree-state-diverging-p inner) t
           (tree-state-alpha-sum inner)
           (+ (tree-state-alpha-sum inner)
              (tree-state-alpha-sum outer))
           (tree-state-n-alpha inner)
           (+ (tree-state-n-alpha inner)
              (tree-state-n-alpha outer)))
     inner)
    (t
     (let* ((combined-log-weight
              (log-sum-exp (tree-state-log-sum-weight inner)
                           (tree-state-log-sum-weight outer)))
            (accept-outer-p
              (and (> (tree-state-log-sum-weight outer)
                      most-negative-double-float)
                   (< (log-random-unit)
                      (- (tree-state-log-sum-weight outer)
                         combined-log-weight)))))
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
       (when accept-outer-p
         (setf (tree-state-q-proposal inner)
               (tree-state-q-proposal outer)
               (tree-state-log-pdf-proposal inner)
               (tree-state-log-pdf-proposal outer)
               (tree-state-grad-proposal inner)
               (tree-state-grad-proposal outer)))
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
             (tree-state-turning-p inner)
             (or (tree-state-turning-p outer)
                 (no-u-turn-p
                  (tree-state-q-minus inner)
                  (tree-state-p-minus inner)
                  (tree-state-q-plus inner)
                  (tree-state-p-plus inner))))
       inner))))

(defun build-tree (log-pdf-fn q p grad log-pdf-q step-size direction depth
                   initial-energy)
  "Recursive balanced binary tree of leapfrog steps for NUTS."
  (if (= depth 0)
      (build-tree-leaf log-pdf-fn q p grad log-pdf-q step-size direction
                       initial-energy)
      (let ((inner (build-tree log-pdf-fn q p grad log-pdf-q
                               step-size direction (1- depth)
                               initial-energy)))
        (if (or (tree-state-diverging-p inner)
                (tree-state-turning-p inner))
            inner
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
              (merge-tree-states inner outer direction))))))

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
  (ensure-valid-parameter
   (and (integerp max-tree-depth) (plusp max-tree-depth) (<= max-tree-depth 25))
   :max-tree-depth
   max-tree-depth
   "nuts: MAX-TREE-DEPTH must be a positive integer <= 25")
  (multiple-value-bind (initial-params step-size)
      (validate-mcmc-common-args "nuts" initial-params n-samples n-warmup step-size)
    (let* ((*log-pdf-error-warned-p* nil)
           (start-time (get-internal-real-time))
           (samples nil)
           (total-iterations (+ n-samples n-warmup))
           (da-state (when (and adapt-step-size (plusp n-warmup))
                       (make-dual-avg-state step-size :target-accept 0.80d0)))
           (sum-accept 0.0d0)
           (n-accept-total 0)
           (n-divergences 0)
           (current-q nil)
           (current-log-pdf nil)
           (current-grad nil))
      (multiple-value-bind (resolved-q resolved-log-pdf resolved-grad)
          (resolve-initial-state log-pdf-fn initial-params :include-gradient t)
        (unless resolved-q
          (return-from nuts
            (values '() 0.0d0 (make-empty-diagnostics n-warmup))))
        (setf current-q resolved-q
              current-log-pdf resolved-log-pdf
              current-grad resolved-grad))
      (let ((n-dim (length current-q)))
        (with-float-traps-masked
          (dotimes (iter total-iterations)
            (let* ((current-p (loop repeat n-dim collect (dist:normal-sample)))
                   (initial-energy (- (compute-kinetic-energy current-p)
                                      current-log-pdf))
                   (q-minus (copy-list current-q))
                   (p-minus (copy-list current-p))
                   (grad-minus (copy-list current-grad))
                   (q-plus (copy-list current-q))
                   (p-plus (copy-list current-p))
                   (grad-plus (copy-list current-grad))
                   (running-log-sum-weight 0.0d0)
                   (iter-alpha-sum 0.0d0)
                   (iter-n-alpha 0))
              (let ((depth 0)
                    (keep-going t)
                    (iter-diverged-p nil))
                (loop while (and keep-going (< depth max-tree-depth))
                      do (let* ((direction (if (< (random 1.0d0) 0.5d0) -1 1))
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
                           (incf iter-alpha-sum (tree-state-alpha-sum tree))
                           (incf iter-n-alpha (tree-state-n-alpha tree))
                           (cond
                             ((tree-state-diverging-p tree)
                              (setf keep-going nil
                                    iter-diverged-p t))
                             (t
                              (if (= direction -1)
                                  (setf q-minus (tree-state-q-minus tree)
                                        p-minus (tree-state-p-minus tree)
                                        grad-minus (tree-state-grad-minus tree))
                                  (setf q-plus (tree-state-q-plus tree)
                                        p-plus (tree-state-p-plus tree)
                                        grad-plus (tree-state-grad-plus tree)))
                              (when (> (tree-state-log-sum-weight tree)
                                       most-negative-double-float)
                                (let ((combined (log-sum-exp
                                                 running-log-sum-weight
                                                 (tree-state-log-sum-weight tree))))
                                  (when (< (log-random-unit)
                                           (- (tree-state-log-sum-weight tree)
                                              combined))
                                    (setf current-q (tree-state-q-proposal tree)
                                          current-log-pdf
                                          (tree-state-log-pdf-proposal tree)
                                          current-grad
                                          (tree-state-grad-proposal tree)))
                                  (setf running-log-sum-weight combined)))
                              (when (or (tree-state-turning-p tree)
                                        (no-u-turn-p q-minus p-minus
                                                     q-plus p-plus))
                                (setf keep-going nil))))
                           (incf depth)))
                (when (and iter-diverged-p (>= iter n-warmup))
                  (incf n-divergences)))
              (incf sum-accept iter-alpha-sum)
              (incf n-accept-total iter-n-alpha))
            (when (and da-state (< iter n-warmup))
              (let ((iter-accept (if (> n-accept-total 0)
                                     (/ sum-accept
                                        (coerce n-accept-total 'double-float))
                                     0.0d0)))
                (setf step-size (dual-avg-update da-state iter-accept))
                (setf sum-accept 0.0d0
                      n-accept-total 0)))
            (when (and da-state (= iter (1- n-warmup)))
              (setf step-size (dual-avg-final-step-size da-state)))
            (when (and (plusp n-warmup) (= iter (1- n-warmup)))
              (setf sum-accept 0.0d0
                    n-accept-total 0))
            (when (>= iter n-warmup)
              (push (copy-list current-q) samples)))))
      (warn-high-divergence n-divergences n-samples)
      (let ((final-accept (if (> n-accept-total 0)
                              (/ sum-accept (coerce n-accept-total 'double-float))
                              0.0d0)))
        (values (nreverse samples)
                final-accept
                (make-final-diagnostics
                 :accept-rate final-accept
                 :n-divergences n-divergences
                 :final-step-size step-size
                 :n-samples n-samples
                 :n-warmup n-warmup
                 :start-time start-time))))))
