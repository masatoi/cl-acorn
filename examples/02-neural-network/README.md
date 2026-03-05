# Neural Network (MLP) Classification with Forward-Mode AD

A multi-layer perceptron (MLP) classifier trained on the Iris dataset using
cl-acorn's forward-mode automatic differentiation for gradient computation.

## What It Demonstrates

- **Neural network training with AD**: All gradients computed via `ad:derivative`
  -- no hand-derived backpropagation formulas needed.
- **Forward-mode cost structure**: Training requires 67 separate `ad:derivative`
  calls per epoch (one per parameter), illustrating why forward-mode AD is
  O(p) for p parameters. Reverse-mode AD (backpropagation) achieves O(1).
- **Composability of AD operations**: Sigmoid activation, matrix multiply,
  softmax cross-entropy loss -- all composed from cl-acorn's overloaded
  arithmetic, with derivatives propagated automatically.

## Architecture

```
Input(4) --> Hidden(8, sigmoid) --> Output(3, softmax) --> Cross-Entropy Loss
```

- **Parameters**: 67 total (W1: 8x4=32, b1: 8, W2: 3x8=24, b2: 3)
- **Activation**: Sigmoid in hidden layer, softmax for output probabilities
- **Loss**: Cross-entropy with log-sum-exp numerical stability trick
- **Optimizer**: Gradient descent (lr=0.5, 100 epochs)

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) (Fisher, 1936):
150 samples, 4 features (sepal/petal length and width), 3 classes (setosa,
versicolor, virginica). A 30-sample subset (10 per class) is used for training
to keep forward-mode runtime tractable.

## How to Run

```lisp
(load "examples/02-neural-network/main.lisp")
```

Expected output: loss decreasing from ~1.1 to ~0.25 over 100 epochs, with
training accuracy reaching ~100% and full-dataset accuracy around 90-98%.

## Key Takeaway

Forward-mode AD computes exact derivatives but scales as O(p) where p is the
number of parameters. For this 67-parameter network, that means 67 forward
passes per gradient step. Real neural networks have millions of parameters,
making reverse-mode AD (backpropagation) essential -- it computes the full
gradient in a single backward pass regardless of parameter count.
