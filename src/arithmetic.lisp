(in-package #:cl-acorn.ad)

;;; --- Binary generic functions ---

(defgeneric binary-add (a b)
  (:documentation "Binary addition supporting dual numbers."))

(defmethod binary-add ((a dual) (b dual))
  (make-dual (cl:+ (dual-real a) (dual-real b))
             (cl:+ (dual-epsilon a) (dual-epsilon b))))

(defmethod binary-add ((a dual) (b number))
  (make-dual (cl:+ (dual-real a) (coerce b 'double-float))
             (dual-epsilon a)))

(defmethod binary-add ((a number) (b dual))
  (make-dual (cl:+ (coerce a 'double-float) (dual-real b))
             (dual-epsilon b)))

(defmethod binary-add ((a number) (b number))
  (cl:+ a b))

(defgeneric binary-sub (a b)
  (:documentation "Binary subtraction supporting dual numbers."))

(defmethod binary-sub ((a dual) (b dual))
  (make-dual (cl:- (dual-real a) (dual-real b))
             (cl:- (dual-epsilon a) (dual-epsilon b))))

(defmethod binary-sub ((a dual) (b number))
  (make-dual (cl:- (dual-real a) (coerce b 'double-float))
             (dual-epsilon a)))

(defmethod binary-sub ((a number) (b dual))
  (make-dual (cl:- (coerce a 'double-float) (dual-real b))
             (cl:- (dual-epsilon b))))

(defmethod binary-sub ((a number) (b number))
  (cl:- a b))

;;; --- N-ary wrappers ---

(defun + (&rest args)
  "Addition supporting dual numbers."
  (case (length args)
    (0 0)
    (1 (first args))
    (t (reduce #'binary-add args))))

(defun - (arg &rest more)
  "Subtraction/negation supporting dual numbers."
  (if more
      (reduce #'binary-sub more :initial-value arg)
      (binary-sub 0 arg)))
