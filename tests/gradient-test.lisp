(defpackage #:cl-acorn/tests/gradient-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/gradient-test)

;;; --- gradient tests ---

(deftest test-gradient-sum
  (testing "gradient of f(x,y) = x + y is (1, 1)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:+ (first p) (second p)))
                     '(3.0d0 4.0d0))
      (ok (approx= val 7.0d0))
      (ok (approx= (first grad) 1.0d0))
      (ok (approx= (second grad) 1.0d0)))))

(deftest test-gradient-product
  (testing "gradient of f(x,y) = x * y is (y, x)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (first p) (second p)))
                     '(3.0d0 4.0d0))
      (ok (approx= val 12.0d0))
      (ok (approx= (first grad) 4.0d0))
      (ok (approx= (second grad) 3.0d0)))))

(deftest test-gradient-polynomial
  (testing "gradient of f(x,y) = x^2 + x*y at (3,4) is (2x+y, x) = (10, 3)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (let ((x (first p)) (y (second p)))
                         (ad:+ (ad:* x x) (ad:* x y))))
                     '(3.0d0 4.0d0))
      (ok (approx= val 21.0d0))
      (ok (approx= (first grad) 10.0d0))
      (ok (approx= (second grad) 3.0d0)))))

(deftest test-gradient-single-variable
  (testing "gradient of f(x) = x^2 at x=3 is (6)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p) (ad:* (first p) (first p)))
                     '(3.0d0))
      (ok (approx= val 9.0d0))
      (ok (approx= (first grad) 6.0d0)))))

(deftest test-gradient-three-variables
  (testing "gradient of f(x,y,z) = x*y*z at (2,3,4) is (yz,xz,xy)=(12,8,6)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (first p) (second p) (third p)))
                     '(2.0d0 3.0d0 4.0d0))
      (ok (approx= val 24.0d0))
      (ok (approx= (first grad) 12.0d0))
      (ok (approx= (second grad) 8.0d0))
      (ok (approx= (third grad) 6.0d0)))))

(deftest test-gradient-transcendental
  (testing "gradient of f(x,y) = sin(x) * exp(y) at (1, 0)"
    ;; df/dx = cos(x)*exp(y) = cos(1)
    ;; df/dy = sin(x)*exp(y) = sin(1)
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (ad:sin (first p)) (ad:exp (second p))))
                     '(1.0d0 0.0d0))
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= (first grad) (cos 1.0d0)))
      (ok (approx= (second grad) (sin 1.0d0))))))

(deftest test-gradient-matches-derivative
  (testing "single-variable gradient matches forward-mode derivative"
    (let ((fn-grad (lambda (p) (ad:exp (ad:sin (first p)))))
          (fn-deriv (lambda (x) (ad:exp (ad:sin x)))))
      (multiple-value-bind (fwd-val fwd-grad) (ad:derivative fn-deriv 1.0d0)
        (multiple-value-bind (rev-val rev-grad) (ad:gradient fn-grad '(1.0d0))
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad (first rev-grad))))))))

(deftest test-gradient-with-loop
  (testing "gradient propagates through iterative computation"
    ;; f(x,y) = sum_{i=0}^{9} (x*y) = 10*x*y
    ;; df/dx = 10y, df/dy = 10x
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (let ((x (first p)) (y (second p))
                             (acc 0.0d0))
                         (dotimes (i 10)
                           (declare (ignore i))
                           (setf acc (ad:+ acc (ad:* x y))))
                         acc))
                     '(2.0d0 3.0d0))
      (ok (approx= val 60.0d0))
      (ok (approx= (first grad) 30.0d0))
      (ok (approx= (second grad) 20.0d0)))))

(deftest test-gradient-integer-params
  (testing "gradient accepts integer parameters"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p) (ad:* (first p) (first p)))
                     '(3))
      (ok (approx= val 9.0d0))
      (ok (approx= (first grad) 6.0d0)))))

;;; --- jacobian-vector-product tests ---

(deftest test-jvp-identity
  (testing "Jvp of f(x) = [x1, x2] with v = [1, 0] gives [1, 0]"
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p) (list (first p) (second p)))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first val) 3.0d0))
      (ok (approx= (second val) 4.0d0))
      (ok (approx= (first jvp) 1.0d0))
      (ok (approx= (second jvp) 0.0d0)))))

(deftest test-jvp-linear
  (testing "Jvp of f(x,y) = [2x+y, x-y] with v = [1,1]"
    ;; J = [[2, 1], [1, -1]], J*v = [3, 0]
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p)
           (let ((x (first p)) (y (second p)))
             (list (ad:+ (ad:* 2 x) y)
                   (ad:- x y))))
         '(1.0d0 1.0d0)
         '(1.0d0 1.0d0))
      (ok (approx= (first val) 3.0d0))
      (ok (approx= (second val) 0.0d0))
      (ok (approx= (first jvp) 3.0d0))
      (ok (approx= (second jvp) 0.0d0)))))

(deftest test-jvp-scalar-function
  (testing "Jvp of scalar f(x,y) = x*y with v = [1,0] gives df/dx"
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p) (list (ad:* (first p) (second p))))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first val) 12.0d0))
      (ok (approx= (first jvp) 4.0d0)))))

;;; --- hessian-vector-product tests ---

(deftest test-hvp-quadratic
  (testing "Hvp of f(x,y) = x^2 + x*y with v = [1, 0]"
    ;; H = [[2, 1], [1, 0]], H*[1,0] = [2, 1]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p)
           (let ((x (first p)) (y (second p)))
             (ad:+ (ad:* x x) (ad:* x y))))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first grad) 10.0d0))   ; df/dx = 2x+y = 10
      (ok (approx= (second grad) 3.0d0))   ; df/dy = x = 3
      (ok (approx= (first hvp) 2.0d0))     ; d2f/dx2*1 + d2f/dxdy*0 = 2
      (ok (approx= (second hvp) 1.0d0))))) ; d2f/dydx*1 + d2f/dy2*0 = 1

(deftest test-hvp-cubic
  (testing "Hvp of f(x) = x^3 with v = [1]"
    ;; f'(x) = 3x^2, f''(x) = 6x
    ;; At x=2: grad = [12], Hvp = [12]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p) (ad:* (first p) (first p) (first p)))
         '(2.0d0)
         '(1.0d0))
      (ok (approx= (first grad) 12.0d0))
      (ok (approx= (first hvp) 12.0d0)))))

(deftest test-hvp-sin
  (testing "Hvp of f(x) = sin(x) with v = [1]"
    ;; f'(x) = cos(x), f''(x) = -sin(x)
    ;; At x=1: grad = [cos(1)], Hvp = [-sin(1)]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p) (ad:sin (first p)))
         '(1.0d0)
         '(1.0d0))
      (ok (approx= (first grad) (cos 1.0d0)))
      (ok (approx= (first hvp) (- (sin 1.0d0)))))))

(deftest test-hvp-multivariate
  (testing "Hvp of f(x,y) = sin(x)*y^2 with v = [1, 1]"
    ;; df/dx = cos(x)*y^2, df/dy = 2*sin(x)*y
    ;; H = [[-sin(x)*y^2, 2*cos(x)*y],
    ;;      [2*cos(x)*y,   2*sin(x)  ]]
    ;; At (1, 2): H = [[-sin(1)*4, 2*cos(1)*2], [2*cos(1)*2, 2*sin(1)]]
    ;; H*[1,1] = [-sin(1)*4+4*cos(1), 4*cos(1)+2*sin(1)]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p)
           (ad:* (ad:sin (first p)) (ad:* (second p) (second p))))
         '(1.0d0 2.0d0)
         '(1.0d0 1.0d0))
      (ok (approx= (first grad) (* (cos 1.0d0) 4.0d0)))
      (ok (approx= (second grad) (* 2.0d0 (sin 1.0d0) 2.0d0)))
      (ok (approx= (first hvp)
                    (+ (* -1.0d0 (sin 1.0d0) 4.0d0)
                       (* 4.0d0 (cos 1.0d0)))))
      (ok (approx= (second hvp)
                    (+ (* 4.0d0 (cos 1.0d0))
                       (* 2.0d0 (sin 1.0d0))))))))
