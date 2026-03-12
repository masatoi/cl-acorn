(defsystem "cl-acorn"
  :version "0.2.0"
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
                 (:file "gradient"))))
  :description "Automatic differentiation using dual numbers and reverse-mode tape"
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
                 (:file "gradient-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
