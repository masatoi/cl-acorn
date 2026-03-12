(in-package #:cl-acorn.ad)

;;; --- sin ---
;;; Forward-over-reverse compatible: uses generic ops on node-values.

(defmethod ad-sin ((x tape-node))
  (let ((v (node-value x)))
    (make-node (ad-sin v)
               (list (cons x (ad-cos v))))))

;;; --- cos ---

(defmethod ad-cos ((x tape-node))
  (let ((v (node-value x)))
    (make-node (ad-cos v)
               (list (cons x (unary-negate (ad-sin v)))))))

;;; --- tan ---

(defmethod ad-tan ((x tape-node))
  (let ((v (node-value x)))
    (let ((cos-v (ad-cos v)))
      (make-node (ad-tan v)
                 (list (cons x (binary-div 1.0d0
                                           (binary-mul cos-v cos-v))))))))

;;; --- exp ---

(defmethod ad-exp ((x tape-node))
  (let ((exp-v (ad-exp (node-value x))))
    (make-node exp-v
               (list (cons x exp-v)))))

;;; --- log (unary) ---

(defmethod unary-log ((x tape-node))
  (let ((v (node-value x)))
    (make-node (unary-log v)
               (list (cons x (binary-div 1.0d0 v))))))

;;; --- sqrt ---

(defmethod ad-sqrt ((x tape-node))
  (let* ((v (node-value x))
         (sqrt-v (ad-sqrt v)))
    (make-node sqrt-v
               (list (cons x (binary-div 1.0d0
                                         (binary-mul 2.0d0 sqrt-v)))))))

;;; --- abs ---

(defmethod ad-abs ((x tape-node))
  (let* ((v (node-value x))
         (real-v (if (typep v 'dual) (dual-real v) v)))
    (make-node (ad-abs v)
               (list (cons x (coerce (cl:signum real-v) 'double-float))))))

;;; --- expt ---

(defmethod binary-expt ((base tape-node) (power number))
  (let ((bv (node-value base))
        (n (coerce power 'double-float)))
    (make-node (binary-expt bv n)
               (list (cons base (binary-mul n
                                            (binary-expt bv (binary-sub n 1.0d0))))))))

(defmethod binary-expt ((base number) (power tape-node))
  (let* ((c (coerce base 'double-float))
         (pv (node-value power))
         (cp (binary-expt c pv)))
    (make-node cp
               (list (cons power (binary-mul cp (unary-log c)))))))

(defmethod binary-expt ((base tape-node) (power tape-node))
  ;; x^y = exp(y * ln(x)) -- reuses AD-aware exp, *, log
  (exp (* power (log base))))
