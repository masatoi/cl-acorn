;;; examples/07-signal-processing/main.lisp --- FIR filter optimization via AD
;;;
;;; Demonstrates using cl-acorn's automatic differentiation to optimize
;;; FIR (Finite Impulse Response) filter coefficients via gradient descent.
;;; A noisy sinusoidal signal is generated, and AD computes the gradient
;;; of the mean squared error with respect to each filter tap, enabling
;;; efficient optimization without manual derivative derivation.

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.signal-processing
  (:use #:cl))

(in-package #:cl-acorn.examples.signal-processing)

;;; ---------------------------------------------------------------------------
;;; Reproducible pseudo-random number generation (LCG + Box-Muller)
;;; ---------------------------------------------------------------------------

(defvar *rng-state* 42
  "State for the linear congruential generator.")

(defun lcg-next ()
  "Advance the LCG and return the new state.
Parameters from glibc: a=1103515245, c=12345, m=2^31."
  (setf *rng-state* (mod (+ (* 1103515245 *rng-state*) 12345)
                         (ash 1 31)))
  *rng-state*)

(defun rng-uniform ()
  "Return a pseudo-random double-float in (0, 1)."
  (/ (float (lcg-next) 1.0d0)
     (float (ash 1 31) 1.0d0)))

(defun rng-gaussian ()
  "Generate a standard normal random value via the Box-Muller transform."
  (let ((u1 (rng-uniform))
        (u2 (rng-uniform)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

;;; ---------------------------------------------------------------------------
;;; Signal generation
;;; ---------------------------------------------------------------------------

(defun generate-signals (n freq sample-rate sigma)
  "Generate a clean sinusoidal signal and a noisy version.
Returns (values clean-vector noisy-vector) as simple-vectors of double-floats.
N samples at SAMPLE-RATE Hz, frequency FREQ Hz, Gaussian noise with SIGMA."
  (setf *rng-state* 42)
  (let ((clean (make-array n :element-type 'double-float))
        (noisy (make-array n :element-type 'double-float)))
    (dotimes (i n)
      (let* ((t-val (/ (float i 1.0d0) (float sample-rate 1.0d0)))
             (clean-val (sin (* 2.0d0 pi freq t-val)))
             (noise (* sigma (rng-gaussian)))
             (noisy-val (+ clean-val noise)))
        (setf (aref clean i) clean-val)
        (setf (aref noisy i) noisy-val)))
    (values clean noisy)))

;;; ---------------------------------------------------------------------------
;;; FIR filter (AD-compatible)
;;; ---------------------------------------------------------------------------

(defun fir-filter (coeffs signal-vec)
  "Apply FIR filter with COEFFS to SIGNAL-VEC using zero-padding.
COEFFS is a list of numbers or dual numbers (one per tap).
SIGNAL-VEC is a simple-vector of double-floats.
Returns a list of filtered values (numbers or duals)."
  (let ((n (length signal-vec))
        (num-taps (length coeffs)))
    (loop for i below n
          collect (let ((acc 0.0d0))
                    (loop for k below num-taps
                          for j = (- i k)
                          do (when (>= j 0)
                               (setf acc
                                     (ad:+ acc
                                           (ad:* (nth k coeffs)
                                                 (aref signal-vec j))))))
                    acc))))

(defun filter-mse (coeffs noisy-vec clean-vec)
  "Mean squared error between FIR-filtered noisy signal and clean reference.
COEFFS may contain dual numbers for AD gradient computation.
Returns a number or dual."
  (let* ((filtered (fir-filter coeffs noisy-vec))
         (n (length filtered))
         (total-error 0.0d0))
    (loop for f in filtered
          for i from 0
          do (let ((diff (ad:- f (aref clean-vec i))))
               (setf total-error (ad:+ total-error (ad:* diff diff)))))
    (ad:/ total-error (float n 1.0d0))))

;;; ---------------------------------------------------------------------------
;;; Plain MSE (no AD, for passthrough comparison)
;;; ---------------------------------------------------------------------------

(defun plain-mse (vec-a vec-b)
  "Mean squared error between two vectors using plain CL arithmetic."
  (let ((n (length vec-a))
        (total 0.0d0))
    (dotimes (i n)
      (let ((diff (- (aref vec-a i) (aref vec-b i))))
        (incf total (* diff diff))))
    (/ total (float n 1.0d0))))

;;; ---------------------------------------------------------------------------
;;; Gradient descent optimization
;;; ---------------------------------------------------------------------------

(defun compute-gradient (coeffs noisy-vec clean-vec)
  "Compute the gradient of MSE with respect to each filter coefficient.
Returns a list of partial derivatives (one per tap)."
  (loop for i below (length coeffs)
        collect (let ((idx i))
                  (nth-value
                   1
                   (ad:derivative
                    (lambda (ci)
                      (let ((cs (copy-list coeffs)))
                        (setf (nth idx cs) ci)
                        (filter-mse cs noisy-vec clean-vec)))
                    (nth idx coeffs))))))

(defun optimize-filter (clean-vec noisy-vec &key (lr 0.01d0) (epochs 200))
  "Optimize 5-tap FIR filter coefficients via gradient descent.
Returns optimized coefficient list.  Prints progress every 20 epochs."
  (let ((coeffs (list 0.2d0 0.2d0 0.2d0 0.2d0 0.2d0)))
    (dotimes (epoch epochs)
      (let ((grads (compute-gradient coeffs noisy-vec clean-vec)))
        ;; Update coefficients
        (setf coeffs (mapcar (lambda (c g) (- c (* lr g)))
                             coeffs grads))
        ;; Print progress
        (when (zerop (mod epoch 20))
          (let ((mse (filter-mse coeffs noisy-vec clean-vec)))
            (format t "  Epoch ~3D: MSE = ~,6F~%" epoch mse)))))
    coeffs))

;;; ---------------------------------------------------------------------------
;;; Output and demonstration
;;; ---------------------------------------------------------------------------

(defun format-coeffs (coeffs)
  "Format a coefficient list as a compact string."
  (format nil "(~{~,4F~^ ~})" coeffs))

(defun run-examples ()
  "Run FIR filter optimization demonstration."
  (let ((freq 5.0d0)
        (sample-rate 100.0d0)
        (n-samples 200)
        (sigma 0.3d0))
    (format t "~&=== FIR Filter Coefficient Optimization via AD ===~%")
    (format t "Signal: ~,1F Hz sine, sample rate ~D Hz, ~D samples, noise sigma=~,1F~%~%"
            freq (round sample-rate) n-samples sigma)

    (multiple-value-bind (clean noisy)
        (generate-signals n-samples freq sample-rate sigma)

      ;; --- Initial assessment ---
      (let* ((passthrough-mse (plain-mse noisy clean))
             (avg-coeffs (list 0.2d0 0.2d0 0.2d0 0.2d0 0.2d0))
             (avg-mse (filter-mse avg-coeffs noisy clean)))
        (format t "--- Initial Assessment ---~%")
        (format t "Passthrough MSE (no filter):  ~,6F~%" passthrough-mse)
        (format t "Averaging filter MSE:         ~,6F~%~%" avg-mse)

        ;; --- Optimization ---
        (format t "--- Gradient Descent Optimization (lr=0.01, 200 epochs) ---~%")
        (let* ((optimized-coeffs (optimize-filter clean noisy :lr 0.01d0 :epochs 200))
               (optimized-mse (filter-mse optimized-coeffs noisy clean)))

          ;; --- Results ---
          (format t "~%--- Results ---~%")
          (format t "Optimized coefficients: ~A~%" (format-coeffs optimized-coeffs))
          (format t "Optimized filter MSE:   ~,6F~%~%" optimized-mse)

          ;; --- Comparison table ---
          (format t "--- Comparison ---~%")
          (format t "~22A  ~10A  ~12A~%" "Method" "MSE" "Improvement")
          (format t "~A~%" (make-string 46 :initial-element #\-))
          (format t "~22A  ~10,6F  ~12A~%"
                  "No filter" passthrough-mse "(baseline)")
          (when (> passthrough-mse 0.0d0)
            (let ((avg-improve (* 100.0d0 (/ (- passthrough-mse avg-mse)
                                             passthrough-mse)))
                  (opt-improve (* 100.0d0 (/ (- passthrough-mse optimized-mse)
                                             passthrough-mse))))
              (format t "~22A  ~10,6F  ~10,1F%~%"
                      "Averaging filter" avg-mse avg-improve)
              (format t "~22A  ~10,6F  ~10,1F%~%"
                      "Optimized filter" optimized-mse opt-improve))))))))

(run-examples)
