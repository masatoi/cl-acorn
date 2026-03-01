(defsystem "cl-acorn"
  :version "0.1.0"
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
                 (:file "interface"))))
  :description "Forward-mode automatic differentiation using dual numbers"
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
                 (:file "derivative-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
