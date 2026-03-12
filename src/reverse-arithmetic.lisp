(in-package #:cl-acorn.ad)

;;; --- Addition: tape-node methods ---

(defmethod binary-add ((a tape-node) (b tape-node))
  (make-node (cl:+ (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b 1.0d0))))

(defmethod binary-add ((a tape-node) (b number))
  (make-node (cl:+ (node-value a) (coerce b 'double-float))
             (list (cons a 1.0d0))))

(defmethod binary-add ((a number) (b tape-node))
  (make-node (cl:+ (coerce a 'double-float) (node-value b))
             (list (cons b 1.0d0))))

;;; --- Subtraction: tape-node methods ---

(defmethod binary-sub ((a tape-node) (b tape-node))
  (make-node (cl:- (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b -1.0d0))))

(defmethod binary-sub ((a tape-node) (b number))
  (make-node (cl:- (node-value a) (coerce b 'double-float))
             (list (cons a 1.0d0))))

(defmethod binary-sub ((a number) (b tape-node))
  (make-node (cl:- (coerce a 'double-float) (node-value b))
             (list (cons b -1.0d0))))

;;; --- Unary negation: tape-node method ---

(defmethod unary-negate ((a tape-node))
  (make-node (cl:- (node-value a))
             (list (cons a -1.0d0))))

;;; --- Multiplication: tape-node methods ---

(defmethod binary-mul ((a tape-node) (b tape-node))
  (make-node (cl:* (node-value a) (node-value b))
             (list (cons a (node-value b))
                   (cons b (node-value a)))))

(defmethod binary-mul ((a tape-node) (b number))
  (let ((bn (coerce b 'double-float)))
    (make-node (cl:* (node-value a) bn)
               (list (cons a bn)))))

(defmethod binary-mul ((a number) (b tape-node))
  (let ((an (coerce a 'double-float)))
    (make-node (cl:* an (node-value b))
               (list (cons b an)))))

;;; --- Division: tape-node methods ---

(defmethod binary-div ((a tape-node) (b tape-node))
  (let ((av (node-value a))
        (bv (node-value b)))
    (make-node (cl:/ av bv)
               (list (cons a (cl:/ 1.0d0 bv))
                     (cons b (cl:/ (cl:- av) (cl:* bv bv)))))))

(defmethod binary-div ((a tape-node) (b number))
  (let ((bn (coerce b 'double-float)))
    (make-node (cl:/ (node-value a) bn)
               (list (cons a (cl:/ 1.0d0 bn))))))

(defmethod binary-div ((a number) (b tape-node))
  (let ((an (coerce a 'double-float))
        (bv (node-value b)))
    (make-node (cl:/ an bv)
               (list (cons b (cl:/ (cl:- an) (cl:* bv bv)))))))

;;; --- Unary reciprocal: tape-node method ---

(defmethod unary-reciprocal ((a tape-node))
  (let ((v (node-value a)))
    (make-node (cl:/ v)
               (list (cons a (cl:/ -1.0d0 (cl:* v v)))))))
