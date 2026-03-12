(in-package #:cl-acorn.ad)

(defvar *tape* nil
  "Dynamic computation graph for reverse-mode AD.
When non-nil, holds a list of tape-node instances in evaluation order
(most recent first). Bound dynamically by GRADIENT and related functions.")

(defclass tape-node ()
  ((value    :initarg :value
             :accessor node-value
             :documentation "Computed value at this node.")
   (gradient :initarg :gradient
             :accessor node-gradient
             :initform 0
             :documentation "Accumulated gradient during backward pass.")
   (children :initarg :children
             :accessor node-children
             :type list
             :initform nil
             :documentation "List of (child-node . local-partial-derivative) pairs."))
  (:documentation "A node in the computation graph for reverse-mode AD.
Each node records its value, its inputs (children), and the local partial
derivative with respect to each input. The backward pass uses these to
accumulate gradients via the chain rule."))

(defun make-node (value children)
  "Create a tape-node with VALUE and CHILDREN, pushing it onto *tape*.
CHILDREN is a list of (tape-node . local-partial) cons cells.
Returns the new node."
  (let ((node (make-instance 'tape-node
                             :value value
                             :children children)))
    (when *tape*
      (push node *tape*))
    node))

(defun backward (output-node)
  "Backpropagate gradients from OUTPUT-NODE through the tape.
Sets OUTPUT-NODE gradient to 1, then traverses *tape* in reverse
evaluation order (which is forward list order), accumulating gradients
via the chain rule.

All node gradients on the tape should be zero before calling this.
Uses generic addition and multiplication to support forward-over-reverse
composition (where gradients may be dual numbers)."
  (setf (node-gradient output-node) 1.0d0)
  (dolist (n *tape*)
    (let ((n-grad (node-gradient n)))
      (when (not (eql n-grad 0))
        (dolist (child-entry (node-children n))
          (let ((child (car child-entry))
                (local-grad (cdr child-entry)))
            (setf (node-gradient child)
                  (binary-add (node-gradient child)
                              (binary-mul n-grad local-grad)))))))))

(defmethod print-object ((n tape-node) stream)
  (print-unreadable-object (n stream :type t)
    (format stream "~A (grad: ~A)" (node-value n) (node-gradient n))))
