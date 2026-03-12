(in-package #:cl-acorn.ad)

(defun gradient (fn params)
  "Compute the gradient of scalar function FN at PARAMS.
FN must accept a list of tape-node values and return a scalar tape-node
or number. PARAMS is a list of numbers.
Returns (values f(params) gradient-list) where gradient-list contains
df/dp_i for each parameter p_i, as double-floats."
  (let* ((*tape* (list t))  ; non-nil to enable recording; t is sentinel
         (input-nodes (mapcar (lambda (p)
                                (make-node (coerce p 'double-float) nil))
                              params))
         (output (funcall fn input-nodes)))
    ;; Remove the sentinel T from the tail of the tape.
    ;; Nodes are pushed to the front, so the sentinel remains last.
    (setf *tape* (nbutlast *tape*))
    (etypecase output
      (tape-node
       (backward output)
       (values (node-value output)
               (mapcar (lambda (n) (coerce (node-gradient n) 'double-float))
                       input-nodes)))
      (number
       (values (coerce output 'double-float)
               (mapcar (constantly 0.0d0) input-nodes))))))
