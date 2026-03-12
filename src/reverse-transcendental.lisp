(in-package #:cl-acorn.ad)

;;; --- sin ---

(defmethod ad-sin ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:sin v)
               (list (cons x (cl:cos v))))))

;;; --- cos ---

(defmethod ad-cos ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:cos v)
               (list (cons x (cl:- (cl:sin v)))))))

;;; --- tan ---

(defmethod ad-tan ((x tape-node))
  (let ((v (node-value x)))
    (let ((cos-v (cl:cos v)))
      (make-node (cl:tan v)
                 (list (cons x (cl:/ 1.0d0 (cl:* cos-v cos-v))))))))

;;; --- exp ---

(defmethod ad-exp ((x tape-node))
  (let ((exp-v (cl:exp (node-value x))))
    (make-node exp-v
               (list (cons x exp-v)))))

;;; --- log (unary) ---

(defmethod unary-log ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:log v)
               (list (cons x (cl:/ 1.0d0 v))))))

;;; --- sqrt ---

(defmethod ad-sqrt ((x tape-node))
  (let* ((v (node-value x))
         (sqrt-v (cl:sqrt v)))
    (make-node sqrt-v
               (list (cons x (cl:/ 1.0d0 (cl:* 2.0d0 sqrt-v)))))))

;;; --- abs ---

(defmethod ad-abs ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:abs v)
               (list (cons x (coerce (cl:signum v) 'double-float))))))

;;; --- expt ---

(defmethod binary-expt ((base tape-node) (power number))
  (let ((bv (node-value base))
        (n (coerce power 'double-float)))
    (make-node (cl:expt bv n)
               (list (cons base (cl:* n (cl:expt bv (cl:- n 1.0d0))))))))

(defmethod binary-expt ((base number) (power tape-node))
  (let* ((c (coerce base 'double-float))
         (pv (node-value power))
         (cp (cl:expt c pv)))
    (make-node cp
               (list (cons power (cl:* cp (cl:log c)))))))

(defmethod binary-expt ((base tape-node) (power tape-node))
  ;; x^y = exp(y * ln(x)) -- reuses AD-aware exp, *, log
  (exp (* power (log base))))
