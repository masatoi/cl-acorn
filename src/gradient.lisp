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

(defun jacobian-vector-product (fn params vector)
  "Compute J*v where J is the Jacobian of FN at PARAMS and v is VECTOR.
FN must accept a list of values and return a list of values (or a single value).
Uses forward-mode: seeds each parameter as dual(param, vector-component).
Returns (values f(params) J*v) where both are lists of double-floats."
  (let* ((dual-inputs (mapcar (lambda (p v)
                                (make-dual p v))
                              params vector))
         (outputs (funcall fn dual-inputs))
         (output-list (if (listp outputs) outputs (list outputs))))
    (values (mapcar (lambda (o)
                      (coerce (if (typep o 'dual) (dual-real o) o)
                              'double-float))
                    output-list)
            (mapcar (lambda (o)
                      (coerce (if (typep o 'dual) (dual-epsilon o) 0)
                              'double-float))
                    output-list))))
