(defpackage #:cl-acorn/tests/distributions-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/distributions-test)

;;; --- log-gammaln tests ---

(deftest test-log-gammaln-integers
  (testing "log-gammaln at integer values matches log((n-1)!)"
    (ok (approx= (dist:log-gammaln 1.0d0) 0.0d0 1d-10))          ; Γ(1) = 0! = 1
    (ok (approx= (dist:log-gammaln 2.0d0) 0.0d0 1d-10))          ; Γ(2) = 1! = 1
    (ok (approx= (dist:log-gammaln 3.0d0) (log 2.0d0) 1d-10))    ; Γ(3) = 2! = 2
    (ok (approx= (dist:log-gammaln 5.0d0) (log 24.0d0) 1d-10))   ; Γ(5) = 4! = 24
    (ok (approx= (dist:log-gammaln 7.0d0) (log 720.0d0) 1d-10)))) ; Γ(7) = 6! = 720

(deftest test-log-gammaln-half
  (testing "log-gammaln at 0.5 equals log(sqrt(pi))"
    (ok (approx= (dist:log-gammaln 0.5d0)
                 (* 0.5d0 (log pi))
                 1d-10))))

;;; --- normal distribution tests ---

(deftest test-normal-log-pdf-standard
  (testing "standard normal log-pdf at known points"
    ;; log N(0|0,1) = -0.5*log(2pi) ~ -0.9189
    (ok (approx= (dist:normal-log-pdf 0.0d0)
                 -0.9189385332046727d0 1d-10))
    ;; log N(1|0,1) = -0.5*log(2pi) - 0.5 ~ -1.4189
    (ok (approx= (dist:normal-log-pdf 1.0d0)
                 -1.4189385332046727d0 1d-10))))

(deftest test-normal-log-pdf-nonstandard
  (testing "normal log-pdf with explicit mu and sigma"
    ;; log N(3|2,0.5): z=(3-2)/0.5=2, -0.5*log(2pi) - log(0.5) - 0.5*4
    (let ((expected (- (- (* -0.5d0 (log (* 2.0d0 pi)))
                          (log 0.5d0))
                       2.0d0)))
      (ok (approx= (dist:normal-log-pdf 3.0d0 :mu 2.0d0 :sigma 0.5d0)
                   expected 1d-10)))))

(deftest test-normal-log-pdf-ad-forward
  (testing "normal log-pdf differentiable via forward-mode"
    ;; d/dx log N(x|0,1) at x=1 = -(x-mu)/sigma^2 = -1.0
    (multiple-value-bind (val deriv)
        (ad:derivative (lambda (x) (dist:normal-log-pdf x)) 1.0d0)
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= deriv -1.0d0 1d-10)))))

(deftest test-normal-log-pdf-ad-reverse
  (testing "normal log-pdf differentiable via reverse-mode (gradient w.r.t. mu)"
    ;; d/dmu log N(1|mu,1) at mu=0 = (x-mu)/sigma^2 = 1.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:normal-log-pdf 1.0d0 :mu (first p) :sigma 1.0d0))
                     '(0.0d0))
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= (first grad) 1.0d0 1d-10)))))

(deftest test-normal-sample-range
  (testing "normal samples have reasonable mean"
    (let* ((n 10000)
           (samples (loop repeat n collect (dist:normal-sample :mu 5.0d0 :sigma 0.1d0)))
           (mean (/ (reduce #'cl:+ samples) n)))
      (ok (approx= mean 5.0d0 0.05d0)))))

;;; --- uniform distribution tests ---

(deftest test-uniform-log-pdf
  (testing "uniform log-pdf at known points"
    ;; Uniform(0,1) at x=0.5: log(1/(1-0)) = 0
    (ok (approx= (dist:uniform-log-pdf 0.5d0) 0.0d0 1d-10))
    ;; Uniform(2,5) at x=3: log(1/(5-2)) = -log(3)
    (ok (approx= (dist:uniform-log-pdf 3.0d0 :low 2.0d0 :high 5.0d0)
                 (- (log 3.0d0)) 1d-10))))

(deftest test-uniform-log-pdf-out-of-bounds
  (testing "uniform log-pdf returns very negative value out of bounds"
    (ok (< (dist:uniform-log-pdf -1.0d0 :low 0.0d0 :high 1.0d0) -1d100))))

(deftest test-uniform-sample-range
  (testing "uniform samples stay within bounds"
    (let ((lo 2.0d0) (hi 5.0d0))
      (loop repeat 1000
            do (let ((s (dist:uniform-sample :low lo :high hi)))
                 (ok (>= s lo))
                 (ok (<= s hi)))))))

;;; --- bernoulli distribution tests ---

(deftest test-bernoulli-log-pdf
  (testing "bernoulli log-pdf at known points"
    ;; Bernoulli(0.7) at x=1: log(0.7)
    (ok (approx= (dist:bernoulli-log-pdf 1.0d0 :prob 0.7d0)
                 (log 0.7d0) 1d-10))
    ;; Bernoulli(0.7) at x=0: log(0.3)
    (ok (approx= (dist:bernoulli-log-pdf 0.0d0 :prob 0.7d0)
                 (log 0.3d0) 1d-10))))

(deftest test-bernoulli-log-pdf-ad
  (testing "bernoulli log-pdf differentiable w.r.t. prob"
    ;; d/dp log Bernoulli(1|p) at p=0.5 = 1/p = 2.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:bernoulli-log-pdf 1.0d0 :prob (first p)))
                     '(0.5d0))
      (ok (approx= val (log 0.5d0) 1d-10))
      (ok (approx= (first grad) 2.0d0 1d-10)))))

(deftest test-bernoulli-sample
  (testing "bernoulli samples have correct mean"
    (let* ((n 10000)
           (samples (loop repeat n collect (dist:bernoulli-sample :prob 0.7d0)))
           (mean (/ (reduce #'+ samples) n)))
      (ok (approx= mean 0.7d0 0.05d0)))))

;;; --- gamma distribution tests ---

(deftest test-gamma-log-pdf
  (testing "gamma log-pdf at known points"
    ;; Gamma(2, 1) at x=1: (2-1)*log(1) - 1*1 + 2*log(1) - logΓ(2) = -1
    (ok (approx= (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 1.0d0)
                 -1.0d0 1d-10))
    ;; Gamma(1, 1) at x=1: Exponential(1) at x=1: -1.0
    (ok (approx= (dist:gamma-log-pdf 1.0d0 :shape 1.0d0 :rate 1.0d0)
                 -1.0d0 1d-10))))

(deftest test-gamma-log-pdf-ad
  (testing "gamma log-pdf differentiable w.r.t. rate"
    ;; d/dr [Gamma(2,r) log-pdf at x=1] = 2/r - 1
    ;; At r=1: 2 - 1 = 1.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate (first p)))
                     '(1.0d0))
      (declare (ignore val))
      (ok (approx= (first grad) 1.0d0 1d-10)))))

(deftest test-gamma-sample-mean
  (testing "gamma samples have correct mean (shape/rate)"
    (let* ((n 10000)
           (k 3.0d0) (r 2.0d0)
           (samples (loop repeat n collect (dist:gamma-sample :shape k :rate r)))
           (mean (/ (reduce #'+ samples) n)))
      ;; Mean = k/r = 1.5
      (ok (approx= mean 1.5d0 0.1d0)))))

;;; --- beta distribution tests ---

(deftest test-beta-log-pdf
  (testing "beta log-pdf at known points"
    ;; Beta(2,3) at x=0.5: pdf = 1.5, log(1.5) ≈ 0.4055
    (let ((expected (log 1.5d0)))
      (ok (approx= (dist:beta-log-pdf 0.5d0 :alpha 2.0d0 :beta 3.0d0)
                   expected 1d-8)))
    ;; Beta(1,1) = Uniform(0,1): log-pdf = 0
    (ok (approx= (dist:beta-log-pdf 0.5d0 :alpha 1.0d0 :beta 1.0d0)
                 0.0d0 1d-10))))

(deftest test-beta-sample-mean
  (testing "beta samples have correct mean (alpha/(alpha+beta))"
    (let* ((n 10000)
           (a 2.0d0) (b 3.0d0)
           (samples (loop repeat n collect (dist:beta-sample :alpha a :beta b)))
           (mean (/ (reduce #'+ samples) n)))
      ;; Mean = 2/5 = 0.4
      (ok (approx= mean 0.4d0 0.05d0)))))

;;; --- poisson distribution tests ---

(deftest test-poisson-log-pdf
  (testing "poisson log-pdf at known points"
    ;; Poisson(5) at k=3: log(e^(-5) * 5^3 / 3!) = log(0.1404) ≈ -1.9635
    (let ((expected (+ (* 3.0d0 (log 5.0d0)) (- 5.0d0) (- (dist:log-gammaln 4.0d0)))))
      (ok (approx= (dist:poisson-log-pdf 3 :rate 5.0d0)
                   expected 1d-10)))
    ;; Poisson(1) at k=0: log(e^(-1)) = -1.0
    (ok (approx= (dist:poisson-log-pdf 0 :rate 1.0d0)
                 -1.0d0 1d-10))))

(deftest test-poisson-log-pdf-ad
  (testing "poisson log-pdf differentiable w.r.t. rate"
    ;; d/dλ [Poisson(λ) log-pdf at k=3] = k/λ - 1
    ;; At λ=5: 3/5 - 1 = -0.4
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:poisson-log-pdf 3 :rate (first p)))
                     '(5.0d0))
      (declare (ignore val))
      (ok (approx= (first grad) -0.4d0 1d-10)))))

(deftest test-poisson-sample-mean
  (testing "poisson samples have correct mean"
    (let* ((n 10000)
           (rate 4.0d0)
           (samples (loop repeat n collect (dist:poisson-sample :rate rate)))
           (mean (/ (reduce #'+ samples) n)))
      (ok (approx= mean rate 0.2d0)))))
