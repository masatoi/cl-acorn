(in-package #:cl-acorn.ad)

;;; --- Unary generic functions ---

(defgeneric ad-sin (x)
  (:documentation "Sine supporting dual numbers."))

(defmethod ad-sin ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:sin a)
               (cl:* (cl:cos a) b))))

(defmethod ad-sin ((x number))
  (cl:sin x))

(defun sin (x)
  "Sine supporting dual numbers."
  (ad-sin x))

(defgeneric ad-cos (x)
  (:documentation "Cosine supporting dual numbers."))

(defmethod ad-cos ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:cos a)
               (cl:- (cl:* (cl:sin a) b)))))

(defmethod ad-cos ((x number))
  (cl:cos x))

(defun cos (x)
  "Cosine supporting dual numbers."
  (ad-cos x))

(defgeneric ad-tan (x)
  (:documentation "Tangent supporting dual numbers."))

(defmethod ad-tan ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((cos-a (cl:cos a)))
      (make-dual (cl:tan a)
                 (cl:/ b (cl:* cos-a cos-a))))))

(defmethod ad-tan ((x number))
  (cl:tan x))

(defun tan (x)
  "Tangent supporting dual numbers."
  (ad-tan x))

(defgeneric ad-exp (x)
  (:documentation "Exponential supporting dual numbers."))

(defmethod ad-exp ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((exp-a (cl:exp a)))
      (make-dual exp-a
                 (cl:* exp-a b)))))

(defmethod ad-exp ((x number))
  (cl:exp x))

(defun exp (x)
  "Exponential supporting dual numbers."
  (ad-exp x))

(defgeneric unary-log (x)
  (:documentation "Natural logarithm supporting dual numbers."))

(defmethod unary-log ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    ;; Guard M5: log undefined for non-positive reals; return sentinel with zero gradient
    (when (<= a 0.0d0)
      (return-from unary-log (make-dual most-negative-double-float 0.0d0)))
    (make-dual (cl:log a)
               (cl:/ b a))))

(defmethod unary-log ((x number))
  (cl:log (coerce x 'double-float)))

(defun log (x &optional base)
  "Logarithm supporting dual numbers and tape nodes. Without base, returns natural log.
Change-of-base: log_b(x) = ln(x) / ln(b). Both x and base may be dual numbers or tape nodes."
  (if base
      (binary-div (unary-log x) (unary-log base))
      (unary-log x)))

(defgeneric ad-sqrt (x)
  (:documentation "Square root supporting dual numbers."))

(defmethod ad-sqrt ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((sqrt-a (cl:sqrt a)))
      (make-dual sqrt-a
                 (cl:/ b (cl:* 2.0d0 sqrt-a))))))

(defmethod ad-sqrt ((x number))
  (cl:sqrt x))

(defun sqrt (x)
  "Square root supporting dual numbers."
  (ad-sqrt x))

(defgeneric ad-abs (x)
  (:documentation "Absolute value supporting dual numbers."))

(defmethod ad-abs ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:abs a)
               (cl:* (coerce (cl:signum a) 'double-float) b))))

(defmethod ad-abs ((x number))
  (cl:abs x))

(defun abs (x)
  "Absolute value supporting dual numbers."
  (ad-abs x))

;;; --- expt (binary) ---

(defgeneric binary-expt (base power)
  (:documentation "Exponentiation supporting dual numbers."))

(defmethod binary-expt ((base dual) (power number))
  (let ((a (dual-real base))
        (b (dual-epsilon base))
        (n (coerce power 'double-float)))
    ;; Special case: x^0 = 1 for all x (including x=0); d/dx[x^0] = 0
    (when (zerop n)
      (return-from binary-expt (make-dual 1.0d0 0.0d0)))
    ;; Guard M4: a=0 with n<=1 -> derivative term 0^(n-1) diverges (n*0^neg*b = NaN)
    ;; n=1: derivative is b (correct); n<1: gradient is +inf, return 0 as sentinel
    (when (and (zerop a) (<= n 1.0d0))
      (return-from binary-expt
        (make-dual 0.0d0 (if (= n 1.0d0) b 0.0d0))))
    (make-dual (cl:expt a n)
               (cl:* n (cl:expt a (cl:- n 1.0d0)) b))))

(defmethod binary-expt ((base number) (power dual))
  (let ((c (coerce base 'double-float))
        (a (dual-real power))
        (b (dual-epsilon power)))
    ;; Special case: 0^0 = 1 by convention; d/da[0^a] at a=0 -> 0 by limit
    (when (and (zerop c) (zerop a))
      (return-from binary-expt (make-dual 1.0d0 0.0d0)))
    ;; Guard: 0^a = 0 for a > 0; derivative 0^a * ln(0) -> 0 by limit
    (when (zerop c)
      (return-from binary-expt (make-dual 0.0d0 0.0d0)))
    ;; Guard M3: negative base -> cl:expt returns complex for non-integer exponent;
    ;; preserve real part, zero gradient (undefined derivative)
    (when (< c 0.0d0)
      (return-from binary-expt
        (make-dual (coerce (cl:realpart (cl:expt c a)) 'double-float) 0.0d0)))
    (let ((ca (cl:expt c a)))
      (make-dual ca
                 (cl:* ca (cl:log c) b)))))

(defmethod binary-expt ((base dual) (power dual))
  ;; x^y = exp(y * ln(x)) -- reuses AD-aware exp, *, log
  (exp (* power (log base))))

(defmethod binary-expt ((base number) (power number))
  (cl:expt base power))

(defun expt (base power)
  "Exponentiation supporting dual numbers."
  (binary-expt base power))
