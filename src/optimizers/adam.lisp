(in-package #:cl-acorn.optimizers)

(defstruct (adam-state (:constructor %make-adam-state))
  "State for the Adam optimizer.
M holds first moment estimates, V holds second moment estimates,
STEP tracks the number of update steps taken."
  (m nil :type list)
  (v nil :type list)
  (step 0 :type fixnum))

(defun make-adam-state (n-params)
  "Create an Adam optimizer state for N-PARAMS parameters.
Initializes moment estimates to zero."
  (%make-adam-state
   :m (make-list n-params :initial-element 0.0d0)
   :v (make-list n-params :initial-element 0.0d0)
   :step 0))

(defun adam-step (params grads state
                  &key (lr 0.001d0) (beta1 0.9d0) (beta2 0.999d0) (epsilon 1d-8))
  "Adam optimizer step. Updates STATE in-place (m, v, step counter).
Returns a new list of updated parameters.
PARAMS and GRADS are lists of numbers of equal length."
  (unless (= (length params) (length grads))
    (error "adam-step: PARAMS and GRADS must have equal length (got ~D and ~D)"
           (length params) (length grads)))
  (unless (= (length params) (length (adam-state-m state)))
    (error "adam-step: PARAMS length must match state dimensionality (got ~D and ~D)"
           (length params) (length (adam-state-m state))))
  (unless (> lr 0)
    (error "adam-step: LR must be positive, got ~S" lr))
  (unless (and (>= beta1 0) (< beta1 1))
    (error "adam-step: BETA1 must be in [0, 1), got ~S" beta1))
  (unless (and (>= beta2 0) (< beta2 1))
    (error "adam-step: BETA2 must be in [0, 1), got ~S" beta2))
  (unless (> epsilon 0)
    (error "adam-step: EPSILON must be positive, got ~S" epsilon))
  (incf (adam-state-step state))
  (let ((t-step (adam-state-step state)))
    ;; Update biased first moment estimate: m <- beta1*m + (1-beta1)*g
    (setf (adam-state-m state)
          (mapcar (lambda (m g) (+ (* beta1 m) (* (- 1.0d0 beta1) g)))
                  (adam-state-m state) grads))
    ;; Update biased second moment estimate: v <- beta2*v + (1-beta2)*g^2
    (setf (adam-state-v state)
          (mapcar (lambda (v g) (+ (* beta2 v) (* (- 1.0d0 beta2) (* g g))))
                  (adam-state-v state) grads))
    ;; Bias correction
    (let ((bc1 (- 1.0d0 (expt beta1 t-step)))
          (bc2 (- 1.0d0 (expt beta2 t-step))))
      ;; Update params: p <- p - lr * m_hat / (sqrt(v_hat) + epsilon)
      (mapcar (lambda (p m v)
                (let ((m-hat (/ m bc1))
                      (v-hat (/ v bc2)))
                  (- p (/ (* lr m-hat) (+ (sqrt v-hat) epsilon)))))
              params (adam-state-m state) (adam-state-v state)))))
