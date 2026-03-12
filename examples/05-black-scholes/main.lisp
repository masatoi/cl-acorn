;;; examples/05-black-scholes/main.lisp --- Black-Scholes option Greeks via AD
;;;
;;; Demonstrates computing all five major option Greeks (Delta, Gamma,
;;; Vega, Theta, Rho) using cl-acorn's automatic differentiation.
;;; Each Greek is a partial derivative of the Black-Scholes call price
;;; with respect to a different input parameter.  Gamma (the second
;;; derivative w.r.t. spot price) uses a central difference on the
;;; AD-computed Delta because the library does not yet support nested
;;; dual numbers (hyper-duals).

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.black-scholes
  (:use #:cl))

(in-package #:cl-acorn.examples.black-scholes)

;;; ---------------------------------------------------------------------------
;;; Normal distribution utilities (AD-compatible)
;;; ---------------------------------------------------------------------------

(defun norm-pdf (x)
  "Standard normal probability density function.
X may be a dual number; all arithmetic uses ad operations."
  (ad:* (ad:/ 1.0d0 (ad:sqrt (ad:* 2.0d0 pi)))
        (ad:exp (ad:* -0.5d0 (ad:* x x)))))

(defun norm-cdf (x)
  "Standard normal cumulative distribution function.
Uses the Abramowitz & Stegun rational approximation (7.1.26).
X may be a dual number; the sign branch uses the real part."
  (let ((p  0.2316419d0)
        (b1 0.319381530d0)
        (b2 -0.356563782d0)
        (b3 1.781477937d0)
        (b4 -1.821255978d0)
        (b5 1.330274429d0))
    (if (if (typep x 'ad:dual) (minusp (ad:dual-real x)) (minusp x))
        ;; N(x) = 1 - N(-x) for x < 0
        (ad:- 1.0d0 (norm-cdf (ad:- x)))
        ;; Direct approximation for x >= 0
        (let* ((tt (ad:/ 1.0d0 (ad:+ 1.0d0 (ad:* p x))))
               (poly (ad:+ (ad:* b1 tt)
                           (ad:* b2 (ad:* tt tt))
                           (ad:* b3 (ad:* tt (ad:* tt tt)))
                           (ad:* b4 (ad:* tt (ad:* tt (ad:* tt tt))))
                           (ad:* b5 (ad:* tt (ad:* tt (ad:* tt (ad:* tt tt))))))))
          (ad:- 1.0d0 (ad:* (norm-pdf x) poly))))))

;;; ---------------------------------------------------------------------------
;;; Black-Scholes call price formula
;;; ---------------------------------------------------------------------------

(defun bs-call-price (spot strike rate vol time-to-expiry)
  "European call option price under the Black-Scholes model.

  C = S*N(d1) - K*exp(-r*T)*N(d2)

where d1 = [log(S/K) + (r + vol^2/2)*T] / (vol*sqrt(T))
      d2 = d1 - vol*sqrt(T)

All arguments may be dual numbers for AD differentiation."
  (let* ((d1 (ad:/ (ad:+ (ad:log (ad:/ spot strike))
                         (ad:* (ad:+ rate
                                      (ad:/ (ad:* vol vol) 2.0d0))
                                time-to-expiry))
                   (ad:* vol (ad:sqrt time-to-expiry))))
         (d2 (ad:- d1 (ad:* vol (ad:sqrt time-to-expiry)))))
    (ad:- (ad:* spot (norm-cdf d1))
          (ad:* strike
                (ad:* (ad:exp (ad:- (ad:* rate time-to-expiry)))
                      (norm-cdf d2))))))

;;; ---------------------------------------------------------------------------
;;; Greeks via automatic differentiation
;;; ---------------------------------------------------------------------------

(defun compute-greeks (s k r v tt)
  "Compute all Black-Scholes Greeks for a European call option.
Returns (values price delta gamma vega theta rho).

S = spot price, K = strike, R = risk-free rate,
V = volatility, TT = time to expiry."
  ;; Delta = dC/dS
  (multiple-value-bind (price delta)
      (ad:derivative (lambda (spot) (bs-call-price spot k r v tt)) s)
    ;; Gamma = d^2C/dS^2 via central difference on the AD delta
    ;; (nested dual numbers are not yet supported)
    (let* ((h 1.0d-5)
           (delta-up (nth-value 1 (ad:derivative
                                   (lambda (s2) (bs-call-price s2 k r v tt))
                                   (cl:+ s h))))
           (delta-down (nth-value 1 (ad:derivative
                                     (lambda (s2) (bs-call-price s2 k r v tt))
                                     (cl:- s h))))
           (gamma (/ (cl:- delta-up delta-down) (cl:* 2.0d0 h))))
      ;; Vega = dC/dvol
      (multiple-value-bind (price2 vega)
          (ad:derivative (lambda (vol) (bs-call-price s k r vol tt)) v)
        (declare (ignore price2))
        ;; Theta = -dC/dT (negative by convention)
        (multiple-value-bind (price3 dc/dt)
            (ad:derivative (lambda (time) (bs-call-price s k r v time)) tt)
          (declare (ignore price3))
          ;; Rho = dC/dr
          (multiple-value-bind (price4 rho)
              (ad:derivative (lambda (rate) (bs-call-price s k rate v tt)) r)
            (declare (ignore price4))
            (values price delta gamma vega (cl:- dc/dt) rho)))))))

;;; ---------------------------------------------------------------------------
;;; Analytical validation (plain CL arithmetic, no dual numbers)
;;; ---------------------------------------------------------------------------

(defun plain-norm-pdf (x)
  "Standard normal PDF using plain CL arithmetic."
  (/ (cl:exp (/ (cl:- (cl:* x x)) 2.0d0))
     (cl:sqrt (cl:* 2.0d0 pi))))

(defun plain-norm-cdf (x)
  "Standard normal CDF using plain CL arithmetic (A&S 7.1.26)."
  (let ((p  0.2316419d0)
        (b1 0.319381530d0)
        (b2 -0.356563782d0)
        (b3 1.781477937d0)
        (b4 -1.821255978d0)
        (b5 1.330274429d0))
    (if (minusp x)
        (cl:- 1.0d0 (plain-norm-cdf (cl:- x)))
        (let* ((tt (/ 1.0d0 (cl:+ 1.0d0 (cl:* p x))))
               (poly (cl:+ (cl:* b1 tt)
                           (cl:* b2 tt tt)
                           (cl:* b3 tt tt tt)
                           (cl:* b4 tt tt tt tt)
                           (cl:* b5 tt tt tt tt tt))))
          (cl:- 1.0d0 (cl:* (plain-norm-pdf x) poly))))))

(defun compute-d1-d2 (s k r v tt)
  "Compute d1 and d2 for the Black-Scholes formula using plain arithmetic.
Returns (values d1 d2)."
  (let* ((d1 (/ (cl:+ (cl:log (/ s k))
                      (cl:* (cl:+ r (/ (cl:* v v) 2.0d0)) tt))
               (cl:* v (cl:sqrt tt))))
         (d2 (cl:- d1 (cl:* v (cl:sqrt tt)))))
    (values d1 d2)))

(defun analytical-delta (s k r v tt)
  "Analytical Delta = N(d1)."
  (plain-norm-cdf (compute-d1-d2 s k r v tt)))

(defun analytical-gamma (s k r v tt)
  "Analytical Gamma = N'(d1) / (S * vol * sqrt(T))."
  (let ((d1 (compute-d1-d2 s k r v tt)))
    (/ (plain-norm-pdf d1)
       (cl:* s v (cl:sqrt tt)))))

(defun analytical-vega (s k r v tt)
  "Analytical Vega = S * N'(d1) * sqrt(T)."
  (let ((d1 (compute-d1-d2 s k r v tt)))
    (cl:* s (plain-norm-pdf d1) (cl:sqrt tt))))

(defun analytical-rho (s k r v tt)
  "Analytical Rho = K * T * exp(-r*T) * N(d2)."
  (multiple-value-bind (d1 d2) (compute-d1-d2 s k r v tt)
    (declare (ignore d1))
    (cl:* k tt (cl:exp (cl:* (cl:- r) tt)) (plain-norm-cdf d2))))

;;; ---------------------------------------------------------------------------
;;; Output formatting and examples
;;; ---------------------------------------------------------------------------

(defun print-greeks-table (s k r v tt)
  "Print a table comparing AD-computed Greeks with analytical values."
  (format t "~&=== Black-Scholes Option Greeks via Automatic Differentiation ===~%")
  (format t "Parameters: S=~,1F  K=~,1F  r=~,2F  vol=~,2F  T=~,1F~%~%"
          s k r v tt)
  (multiple-value-bind (price delta gamma vega theta rho)
      (compute-greeks s k r v tt)
    (format t "Call price: ~,6F~%~%" price)
    (format t "~12A  ~14A  ~14A  ~12A~%"
            "Greek" "AD Value" "Analytical" "Error")
    (format t "~A~%" (make-string 56 :initial-element #\-))
    ;; Delta
    (let ((a-delta (analytical-delta s k r v tt)))
      (format t "~12A  ~14,8F  ~14,8F  ~12,2E~%"
              "Delta" delta a-delta (cl:abs (cl:- delta a-delta))))
    ;; Gamma
    (let ((a-gamma (analytical-gamma s k r v tt)))
      (format t "~12A  ~14,8F  ~14,8F  ~12,2E~%"
              "Gamma" gamma a-gamma (cl:abs (cl:- gamma a-gamma))))
    ;; Vega
    (let ((a-vega (analytical-vega s k r v tt)))
      (format t "~12A  ~14,8F  ~14,8F  ~12,2E~%"
              "Vega" vega a-vega (cl:abs (cl:- vega a-vega))))
    ;; Theta (no standard closed-form; skip analytical)
    (format t "~12A  ~14,8F  ~14A  ~12A~%"
            "Theta" theta "---" "---")
    ;; Rho
    (let ((a-rho (analytical-rho s k r v tt)))
      (format t "~12A  ~14,8F  ~14,8F  ~12,2E~%"
              "Rho" rho a-rho (cl:abs (cl:- rho a-rho))))))

(defun print-delta-vs-spot (k r v tt)
  "Print Delta as a function of spot price from 80 to 120."
  (format t "~&~%=== Delta vs Spot Price ===~%")
  (format t "K=~,1F  r=~,2F  vol=~,2F  T=~,1F~%~%" k r v tt)
  (format t "~10A  ~14A  ~14A~%"
          "Spot" "Call Price" "Delta")
  (format t "~A~%" (make-string 42 :initial-element #\-))
  (loop for spot from 80.0d0 to 120.0d0 by 5.0d0
        do (multiple-value-bind (price delta)
               (ad:derivative (lambda (s) (bs-call-price s k r v tt)) spot)
             (format t "~10,1F  ~14,6F  ~14,6F~%"
                     spot price delta))))

(defun run-examples ()
  "Run all Black-Scholes demonstrations."
  (let ((s 100.0d0) (k 100.0d0) (r 0.05d0) (v 0.2d0) (tt 1.0d0))
    (print-greeks-table s k r v tt)
    (print-delta-vs-spot k r v tt)))

(run-examples)
