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

(defgeneric unary-negate (a)
  (:documentation "Unary negation supporting dual numbers.
Computes -(a + b*eps) = -a - b*eps."))

(defmethod unary-negate ((a dual))
  (make-dual (cl:- (dual-real a))
             (cl:- (dual-epsilon a))))

(defmethod unary-negate ((a number))
  (cl:- a))

(defun - (arg &rest more)
  "Subtraction/negation supporting dual numbers."
  (if more
      (reduce #'binary-sub more :initial-value arg)
      (unary-negate arg)))

;;; --- Multiplication ---

(defgeneric binary-mul (a b)
  (:documentation "Binary multiplication supporting dual numbers."))

(defmethod binary-mul ((a dual) (b dual))
  (let ((ar (dual-real a))
        (ae (dual-epsilon a))
        (br (dual-real b))
        (be (dual-epsilon b)))
    (make-dual (cl:* ar br)
               (cl:+ (cl:* ar be) (cl:* ae br)))))

(defmethod binary-mul ((a dual) (b number))
  (let ((n (coerce b 'double-float)))
    (make-dual (cl:* (dual-real a) n)
               (cl:* (dual-epsilon a) n))))

(defmethod binary-mul ((a number) (b dual))
  (let ((n (coerce a 'double-float)))
    (make-dual (cl:* n (dual-real b))
               (cl:* n (dual-epsilon b)))))

(defmethod binary-mul ((a number) (b number))
  (cl:* a b))

;;; --- Division ---

(defgeneric binary-div (a b)
  (:documentation "Binary division supporting dual numbers."))

(defmethod binary-div ((a dual) (b dual))
  (let ((p (dual-real a))
        (q (dual-epsilon a))
        (c (dual-real b))
        (d (dual-epsilon b)))
    (let ((c2 (cl:* c c)))
      (make-dual (cl:/ p c)
                 (cl:/ (cl:- (cl:* q c) (cl:* p d))
                       c2)))))

(defmethod binary-div ((a dual) (b number))
  (let ((n (coerce b 'double-float)))
    (make-dual (cl:/ (dual-real a) n)
               (cl:/ (dual-epsilon a) n))))

(defmethod binary-div ((a number) (b dual))
  (let ((p (coerce a 'double-float))
        (c (dual-real b))
        (d (dual-epsilon b)))
    (make-dual (cl:/ p c)
               (cl:/ (cl:- (cl:* p d))
                     (cl:* c c)))))

(defmethod binary-div ((a number) (b number))
  (cl:/ a b))

;;; --- N-ary wrappers (mul/div) ---

(defun * (&rest args)
  "Multiplication supporting dual numbers."
  (case (length args)
    (0 1)
    (1 (first args))
    (t (reduce #'binary-mul args))))

(defgeneric unary-reciprocal (a)
  (:documentation "Unary reciprocal supporting dual numbers.
Computes 1/(a + b*eps) = 1/a - b/a^2 * eps."))

(defmethod unary-reciprocal ((a dual))
  (let ((r (dual-real a))
        (e (dual-epsilon a)))
    (make-dual (cl:/ r)
               (cl:/ (cl:- e) (cl:* r r)))))

(defmethod unary-reciprocal ((a number))
  (cl:/ a))

(defun / (arg &rest more)
  "Division/reciprocal supporting dual numbers."
  (if more
      (reduce #'binary-div more :initial-value arg)
      (unary-reciprocal arg)))
