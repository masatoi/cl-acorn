(in-package #:cl-acorn.ad)

;;; --- Addition: tape-node methods ---
;;; Forward-over-reverse compatible: uses generic ops on node-values
;;; so that dual-number values propagate correctly.

(defmethod binary-add ((a tape-node) (b tape-node))
  (make-node (binary-add (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b 1.0d0))))

(defmethod binary-add ((a tape-node) (b number))
  (make-node (binary-add (node-value a) b)
             (list (cons a 1.0d0))))

(defmethod binary-add ((a number) (b tape-node))
  (make-node (binary-add a (node-value b))
             (list (cons b 1.0d0))))

;;; --- Subtraction: tape-node methods ---

(defmethod binary-sub ((a tape-node) (b tape-node))
  (make-node (binary-sub (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b -1.0d0))))

(defmethod binary-sub ((a tape-node) (b number))
  (make-node (binary-sub (node-value a) b)
             (list (cons a 1.0d0))))

(defmethod binary-sub ((a number) (b tape-node))
  (make-node (binary-sub a (node-value b))
             (list (cons b -1.0d0))))

;;; --- Unary negation: tape-node method ---

(defmethod unary-negate ((a tape-node))
  (make-node (unary-negate (node-value a))
             (list (cons a -1.0d0))))

;;; --- Multiplication: tape-node methods ---

(defmethod binary-mul ((a tape-node) (b tape-node))
  (make-node (binary-mul (node-value a) (node-value b))
             (list (cons a (node-value b))
                   (cons b (node-value a)))))

(defmethod binary-mul ((a tape-node) (b number))
  (make-node (binary-mul (node-value a) b)
             (list (cons a b))))

(defmethod binary-mul ((a number) (b tape-node))
  (make-node (binary-mul a (node-value b))
             (list (cons b a))))

;;; --- Division: tape-node methods ---

(defmethod binary-div ((a tape-node) (b tape-node))
  (let ((av (node-value a))
        (bv (node-value b)))
    (make-node (binary-div av bv)
               (list (cons a (binary-div 1.0d0 bv))
                     (cons b (binary-div (unary-negate av)
                                         (binary-mul bv bv)))))))

(defmethod binary-div ((a tape-node) (b number))
  (make-node (binary-div (node-value a) b)
             (list (cons a (binary-div 1.0d0 b)))))

(defmethod binary-div ((a number) (b tape-node))
  (let ((bv (node-value b)))
    (make-node (binary-div a bv)
               (list (cons b (binary-div (unary-negate a)
                                         (binary-mul bv bv)))))))

;;; --- Unary reciprocal: tape-node method ---

(defmethod unary-reciprocal ((a tape-node))
  (let ((v (node-value a)))
    (make-node (unary-reciprocal v)
               (list (cons a (binary-div -1.0d0 (binary-mul v v)))))))
