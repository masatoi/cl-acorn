(in-package #:cl-acorn.optimizers)

(defun sgd-step (params grads &key (lr 0.01d0))
  "Stochastic gradient descent step: p_i <- p_i - lr * g_i.
PARAMS and GRADS are lists of numbers of equal length.
Returns a new list of updated parameters (no side effects)."
  (unless (= (length params) (length grads))
    (error "sgd-step: PARAMS and GRADS must have equal length (got ~D and ~D)"
           (length params) (length grads)))
  (unless (> lr 0)
    (error "sgd-step: LR must be positive, got ~S" lr))
  (mapcar (lambda (p g) (- p (* lr g))) params grads))
