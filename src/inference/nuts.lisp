(in-package #:cl-acorn.inference)

(defvar +max-delta-energy+ 1000.0d0
  "Maximum energy difference before declaring divergence.")

(defun log-sum-exp (a b)
  "Numerically stable log(exp(a) + exp(b))."
  (let ((m (max a b)))
    (if (= m most-negative-double-float)
        most-negative-double-float
        (+ m (log (+ (exp (- a m)) (exp (- b m))))))))

(defun dot-product (a b)
  "Dot product of two lists of numbers."
  (reduce #'+ (mapcar #'* a b)))

(defun no-u-turn-p (q-minus p-minus q-plus p-plus)
  "Check the no-U-turn criterion.
Returns true if the trajectory should stop (U-turn detected)."
  (let ((dq (mapcar #'- q-plus q-minus)))
    (or (<= (dot-product dq p-plus) 0.0d0)
        (<= (dot-product dq p-minus) 0.0d0))))

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
        (let* ((kinetic (compute-kinetic-energy p-new))
               (new-energy (- kinetic log-pdf-new))
               (delta-energy (- new-energy initial-energy))
               (diverging (> delta-energy +max-delta-energy+))
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
           :alpha-sum accept-stat
           :n-alpha 1)))
      ;; Recursive case: build two halves
      (let ((inner (build-tree log-pdf-fn q p grad log-pdf-q
                               step-size direction (1- depth)
                               initial-energy)))
        (if (tree-state-diverging-p inner)
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
              (if (tree-state-diverging-p outer)
                  ;; Keep inner's proposal but mark as diverging
                  (progn
                    (setf (tree-state-diverging-p inner) t
                          (tree-state-alpha-sum inner)
                          (+ (tree-state-alpha-sum inner)
                             (tree-state-alpha-sum outer))
                          (tree-state-n-alpha inner)
                          (+ (tree-state-n-alpha inner)
                             (tree-state-n-alpha outer)))
                    inner)
                  ;; Combine the two halves
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
                    ;; Combine statistics
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
                          ;; Check U-turn on combined trajectory
                          (tree-state-diverging-p inner)
                          (no-u-turn-p
                           (tree-state-q-minus inner)
                           (tree-state-p-minus inner)
                           (tree-state-q-plus inner)
                           (tree-state-p-plus inner)))
                    inner)))))))

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
Returns (values samples accept-rate)."
  (assert (and (listp initial-params) (plusp (length initial-params))) nil
          "nuts: INITIAL-PARAMS must be a non-empty list")
  (assert (and (integerp n-samples) (plusp n-samples)) nil
          "nuts: N-SAMPLES must be a positive integer")
  (assert (and (integerp n-warmup) (not (minusp n-warmup))) nil
          "nuts: N-WARMUP must be a non-negative integer")
  (assert (> step-size 0.0d0) nil "nuts: STEP-SIZE must be a positive number")
  (assert (and (integerp max-tree-depth) (plusp max-tree-depth)) nil
          "nuts: MAX-TREE-DEPTH must be a positive integer")
  (let* ((current-q (mapcar (lambda (x) (coerce x 'double-float)) initial-params))
         (n-dim (length initial-params))
         (samples nil)
         (total-iterations (+ n-samples n-warmup))
         (step-size (coerce step-size 'double-float))
         (da-state (when (and adapt-step-size (plusp n-warmup))
                     (make-dual-avg-state step-size :target-accept 0.80d0)))
         (sum-accept 0.0d0)
         (n-accept-total 0))
    ;; Get initial log-pdf and gradient
    (multiple-value-bind (init-val init-grad)
        (ad:gradient log-pdf-fn current-q)
      (let ((current-log-pdf (coerce init-val 'double-float))
            (current-grad init-grad))
        (dotimes (iter total-iterations)
          ;; Sample random momentum
          (let* ((current-p (loop repeat n-dim collect (dist:normal-sample)))
                 (initial-energy (- (compute-kinetic-energy current-p)
                                    current-log-pdf)))
            ;; Build tree by doubling
            (let ((depth 0)
                  (keep-going t))
              (loop while (and keep-going (< depth max-tree-depth))
                    do (let* ((direction (if (< (random 1.0d0) 0.5d0) -1 1))
                              (tree (build-tree log-pdf-fn
                                                current-q current-p current-grad
                                                current-log-pdf
                                                step-size direction depth
                                                initial-energy)))
                         ;; Accept proposal if tree produced valid states
                         (when (and (not (tree-state-diverging-p tree))
                                    (> (tree-state-n-valid tree) 0))
                           (setf current-q (tree-state-q-proposal tree)
                                 current-log-pdf (tree-state-log-pdf-proposal tree)
                                 current-grad (tree-state-grad-proposal tree)))
                         ;; Track acceptance statistics
                         (incf sum-accept (tree-state-alpha-sum tree))
                         (incf n-accept-total (tree-state-n-alpha tree))
                         ;; Check stopping criteria
                         (when (tree-state-diverging-p tree)
                           (setf keep-going nil))
                         (incf depth)))))
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
          ;; Collect sample after warmup
          (when (>= iter n-warmup)
            (push (copy-list current-q) samples)))))
    ;; Compute average accept rate over sampling phase
    (let ((final-accept (if (> n-accept-total 0)
                            (/ sum-accept (coerce n-accept-total 'double-float))
                            0.0d0)))
      (values (nreverse samples) final-accept))))
