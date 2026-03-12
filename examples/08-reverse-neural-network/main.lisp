;;;; main.lisp --- MLP classifier on Iris using reverse-mode AD
;;;;
;;;; Demonstrates training a multi-layer perceptron (MLP) neural network
;;;; using cl-acorn's reverse-mode automatic differentiation (ad:gradient).
;;;;
;;;; Architecture: Input(4) -> Hidden(8, sigmoid) -> Output(3, softmax)
;;;; Dataset: Iris (full 150-sample dataset)
;;;;
;;;; Key difference from example 02 (forward-mode):
;;;;   Forward-mode required 67 separate ad:derivative calls per epoch,
;;;;   each computing one partial derivative. Reverse-mode computes all
;;;;   67 partial derivatives in a single ad:gradient call.
;;;;
;;;; Usage:
;;;;   (load "examples/08-reverse-neural-network/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.reverse-neural-network
  (:use #:cl)
  (:export #:sigmoid
           #:forward-pass
           #:softmax-cross-entropy
           #:total-loss
           #:predict
           #:accuracy
           #:init-params
           #:train
           #:run-examples))

(in-package #:cl-acorn.examples.reverse-neural-network)

;;; --------------------------------------------------------------------------
;;; Load Iris dataset from example 02
;;; --------------------------------------------------------------------------

(load (merge-pathnames "../02-neural-network/data.lisp" *load-pathname*))

;;; --------------------------------------------------------------------------
;;; Network constants
;;; --------------------------------------------------------------------------

(defconstant +input-size+ 4)
(defconstant +hidden-size+ 8)
(defconstant +output-size+ 3)
(defconstant +num-params+ (+ (* +hidden-size+ +input-size+)  ; W1: 8x4 = 32
                              +hidden-size+                   ; b1: 8
                              (* +output-size+ +hidden-size+) ; W2: 3x8 = 24
                              +output-size+)                  ; b2: 3 = 67
  "Total number of trainable parameters.")

;;; --------------------------------------------------------------------------
;;; Activation function
;;; --------------------------------------------------------------------------

(defun sigmoid (x)
  "Sigmoid activation: 1/(1 + exp(-x)).
X may be a dual number, tape-node, or plain number."
  (ad:/ 1.0d0 (ad:+ 1.0d0 (ad:exp (ad:- x)))))

;;; --------------------------------------------------------------------------
;;; Forward pass
;;; --------------------------------------------------------------------------

(defun param-ref (params index)
  "Access parameter at INDEX from the flat parameter list."
  (nth index params))

(defun forward-pass (params input)
  "Compute forward pass through the MLP.
PARAMS is a flat list of 67 values (numbers, duals, or tape-nodes).
INPUT is a list of 4 feature values.
Returns a list of 3 output logits.

Parameter layout:
  [0..31]   W1 (8x4, row-major: neuron i, input j -> i*4+j)
  [32..39]  b1 (8 biases)
  [40..63]  W2 (3x8, row-major: output i, hidden j -> 40+i*8+j)
  [64..66]  b2 (3 biases)"
  (let ((hidden (make-list +hidden-size+ :initial-element 0.0d0))
        (output (make-list +output-size+ :initial-element 0.0d0)))
    ;; Hidden layer: h_i = sigmoid(sum_j(W1[i,j] * input[j]) + b1[i])
    (dotimes (i +hidden-size+)
      (let ((acc (param-ref params (+ (* +hidden-size+ +input-size+) i)))) ; b1[i]
        (dotimes (j +input-size+)
          (setf acc (ad:+ acc (ad:* (param-ref params (+ (* i +input-size+) j))
                                    (nth j input)))))
        (setf (nth i hidden) (sigmoid acc))))
    ;; Output layer: o_i = sum_j(W2[i,j] * hidden[j]) + b2[i]
    (let ((w2-offset (* +hidden-size+ +input-size+)))
      (setf w2-offset (+ w2-offset +hidden-size+)) ; skip past W1 and b1
      (dotimes (i +output-size+)
        (let ((acc (param-ref params (+ w2-offset
                                        (* +output-size+ +hidden-size+)
                                        i)))) ; b2[i]
          (dotimes (j +hidden-size+)
            (setf acc (ad:+ acc (ad:* (param-ref params
                                                 (+ w2-offset (* i +hidden-size+) j))
                                      (nth j hidden)))))
          (setf (nth i output) acc))))
    output))

(defun real-value (x)
  "Extract a plain double-float from X.
Handles tape-nodes (reverse-mode), duals (forward-mode), and plain numbers."
  (typecase x
    (ad:tape-node (coerce (ad:node-value x) 'double-float))
    (ad:dual (ad:dual-real x))
    (t (coerce x 'double-float))))

;;; --------------------------------------------------------------------------
;;; Loss function
;;; --------------------------------------------------------------------------

(defun softmax-cross-entropy (logits label)
  "Cross-entropy loss for a single sample.
LOGITS is a list of 3 values (numbers, duals, or tape-nodes).
LABEL is an integer 0, 1, or 2.
Uses the log-sum-exp trick for numerical stability."
  ;; Find max logit (using real parts for comparison)
  (let ((max-val (real-value (first logits))))
    (dolist (logit (rest logits))
      (let ((rv (real-value logit)))
        (when (> rv max-val)
          (setf max-val rv))))
    ;; Compute log(sum(exp(logit_i - max))) using ad arithmetic
    ;; The max-val is a plain number, subtracted from dual/tape logits
    (let ((sum-exp 0.0d0))
      (dolist (logit logits)
        (setf sum-exp (ad:+ sum-exp (ad:exp (ad:- logit max-val)))))
      ;; loss = -(logit[label] - max) + log(sum_exp)
      (ad:+ (ad:- (ad:- (nth label logits) max-val))
            (ad:log sum-exp)))))

(defun total-loss (params features labels)
  "Average cross-entropy loss over all samples.
PARAMS is a flat list of 67 parameters (numbers or tape-nodes).
FEATURES is a list of feature sublists.
LABELS is a list of integer class labels."
  (let ((n (length features))
        (acc 0.0d0))
    (loop for feat in features
          for lab in labels
          do (let ((logits (forward-pass params feat)))
               (setf acc (ad:+ acc (softmax-cross-entropy logits lab)))))
    (ad:/ acc (coerce n 'double-float))))

;;; --------------------------------------------------------------------------
;;; Prediction and evaluation
;;; --------------------------------------------------------------------------

(defun predict (params input)
  "Return predicted class (0, 1, or 2) as argmax of output logits."
  (let ((logits (forward-pass params input))
        (best-class 0)
        (best-val most-negative-double-float))
    (loop for i from 0 below +output-size+
          for logit in logits
          do (let ((rv (real-value logit)))
               (when (> rv best-val)
                 (setf best-val rv)
                 (setf best-class i))))
    best-class))

(defun accuracy (params features labels)
  "Classification accuracy (0.0 to 1.0) on the given dataset."
  (let ((correct 0)
        (total 0))
    (loop for feat in features
          for lab in labels
          do (when (= (predict params feat) lab)
               (incf correct))
             (incf total))
    (/ (coerce correct 'double-float) (coerce total 'double-float))))

;;; --------------------------------------------------------------------------
;;; Parameter initialization
;;; --------------------------------------------------------------------------

(defun init-params ()
  "Initialize network parameters uniformly in [-0.1, 0.1]."
  (loop for i below +num-params+
        collect (- (random 0.2d0) 0.1d0)))

;;; --------------------------------------------------------------------------
;;; Training (reverse-mode)
;;; --------------------------------------------------------------------------

(defun train (features labels &key (lr 0.1d0) (epochs 200))
  "Train the MLP using gradient descent with reverse-mode AD.
One ad:gradient call per epoch computes all 67 partial derivatives
simultaneously -- O(1) backward passes regardless of parameter count.
Returns the trained parameter list."
  (let ((params (init-params)))
    (dotimes (epoch epochs)
      (multiple-value-bind (loss grads)
          (ad:gradient
           (lambda (p) (total-loss p features labels))
           params)
        ;; Update parameters
        (setf params
              (mapcar (lambda (p g) (- p (* lr g)))
                      params grads))
        ;; Report every 10 epochs
        (when (zerop (mod epoch 10))
          (format t "  Epoch ~3D  |  Loss ~8,4F  |  Accuracy ~5,1F%~%"
                  epoch loss
                  (* 100.0d0 (accuracy params features labels))))))
    params))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-examples ()
  "Train an MLP on Iris using reverse-mode AD and report results."
  (format t "~%================================================~%")
  (format t " Neural Network (MLP) with Reverse-Mode AD~%")
  (format t "================================================~%~%")
  (format t "Architecture: Input(4) -> Hidden(8, sigmoid) -> Output(3, softmax)~%")
  (format t "Parameters:   ~D total (W1:32 + b1:8 + W2:24 + b2:3)~%" +num-params+)
  (format t "Dataset:      Iris (full 150 samples)~%")
  (format t "Optimizer:    SGD (lr=0.1, epochs=200)~%~%")
  (format t "Reverse-mode AD computes all ~D gradients in a single backward pass.~%" +num-params+)
  (format t "(Forward-mode example 02 needed ~D separate derivative calls per epoch.)~%~%" +num-params+)

  (format t "--- Training (150 samples, ~D params) ---~%~%" +num-params+)
  (let ((params (train *iris-features* *iris-labels* :lr 0.1d0 :epochs 200)))
    (let ((acc (accuracy params *iris-features* *iris-labels*)))
      (format t "~%--- Results ---~%~%")
      (format t "  Accuracy (150 samples): ~5,1F%~%" (* 100.0d0 acc))
      (format t "~%Key takeaway: Reverse-mode AD computes all gradients in O(1)~%")
      (format t "backward passes, making it practical for models with many parameters.~%"))))

(run-examples)
