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
    ;; Guard M5: log undefined for non-positive reals; return sentinel with zero gradient
    (let ((real-v (if (typep v 'dual) (dual-real v) v)))
      (when (<= real-v 0.0d0)
        (return-from unary-log
          (make-node most-negative-double-float (list (cons x 0.0d0))))))
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
    ;; Special case: x^0 = 1 for all x (including x=0); d/dx[x^0] = 0
    (when (zerop n)
      (return-from binary-expt
        (make-node 1.0d0 (list (cons base 0.0d0)))))
    ;; Guard M4: bv=0 with n<=1 -> gradient n*0^(n-1) diverges to NaN
    ;; n=1: gradient is 1 (correct); n<1: gradient is +inf, return 0 as sentinel
    (let ((real-bv (if (typep bv 'dual) (dual-real bv) bv)))
      (when (and (zerop real-bv) (<= n 1.0d0))
        (return-from binary-expt
          (make-node 0.0d0 (list (cons base (if (= n 1.0d0) 1.0d0 0.0d0)))))))
    (make-node (binary-expt bv n)
               (list (cons base (binary-mul n
                                            (binary-expt bv (binary-sub n 1.0d0))))))))

(defmethod binary-expt ((base number) (power tape-node))
  (let* ((c (coerce base 'double-float)))
    ;; Special case: 0^0 = 1 by convention; d/da[0^a] at a=0 -> 0 by limit
    (let ((real-pv (let ((pv (node-value power)))
                    (if (typep pv 'dual) (dual-real pv) pv))))
      (when (and (zerop c) (zerop real-pv))
        (return-from binary-expt
          (make-node 1.0d0 (list (cons power 0.0d0))))))
    ;; Guard: 0^a = 0 for a > 0; derivative is 0 by limit
    (when (zerop c)
      (return-from binary-expt
        (make-node 0.0d0 (list (cons power 0.0d0)))))
    ;; Guard M3: negative base -> cl:expt returns complex for non-integer exponent;
    ;; preserve real part, zero gradient (undefined derivative)
    (when (< c 0.0d0)
      (let* ((pv (node-value power))
             (real-val (coerce (cl:realpart (cl:expt c pv)) 'double-float)))
        (return-from binary-expt
          (make-node real-val (list (cons power 0.0d0))))))
    (let* ((pv (node-value power))
           (cp (binary-expt c pv)))
      (make-node cp
                 (list (cons power (binary-mul cp (unary-log c))))))))

(defmethod binary-expt ((base tape-node) (power tape-node))
  ;; x^y = exp(y * ln(x)) -- reuses AD-aware exp, *, log
  (exp (* power (log base))))
