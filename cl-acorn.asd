(defsystem "cl-acorn"
  :version "0.3.0"
  :author ""
  :license "MIT"
  :depends-on ()
  :components ((:module "src"
                :serial t
                :components
                ((:file "package")
                 (:file "dual")
                 (:file "arithmetic")
                 (:file "transcendental")
                 (:file "interface")
                 (:file "tape")
                 (:file "reverse-arithmetic")
                 (:file "reverse-transcendental")
                 (:file "gradient")
                 (:module "distributions"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "util")
                   (:file "normal")
                   (:file "uniform")
                   (:file "bernoulli")
                   (:file "gamma")
                   (:file "beta")
                   (:file "poisson")))
                 (:module "optimizers"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "sgd")
                   (:file "adam")))
                 (:module "inference"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "conditions")
                   (:file "dual-avg")
                   (:file "hmc")
                   (:file "nuts")
                   (:file "vi"))))))
  :description "Automatic differentiation and probabilistic inference building blocks"
  :in-order-to ((test-op (test-op "cl-acorn/tests"))))

(defsystem "cl-acorn/tests"
  :author ""
  :license "MIT"
  :depends-on ("cl-acorn"
               "rove")
  :components ((:module "tests"
                :serial t
                :components
                ((:file "util")
                 (:file "dual-test")
                 (:file "arithmetic-test")
                 (:file "transcendental-test")
                 (:file "derivative-test")
                 (:file "tape-test")
                 (:file "reverse-arithmetic-test")
                 (:file "reverse-transcendental-test")
                 (:file "gradient-test")
                 (:file "distributions-test")
                 (:file "optimizers-test")
                 (:file "hmc-test")
                 (:file "nuts-test")
                 (:file "vi-test")
                 (:file "validation-test")
                 (:file "conditions-test")
                 (:file "inference-diagnostics-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-acorn/benchmarks"
  :author ""
  :license "MIT"
  :depends-on ("cl-acorn")
  :components ((:module "benchmarks/cl"
                :serial t
                :components
                ((:file "package")
                 (:file "bench-utils")
                 (:file "bench-ad")
                 (:file "bench-distributions")
                 (:file "bench-inference")
                 (:file "run-all"))))
  :description "Benchmark suite for cl-acorn")
