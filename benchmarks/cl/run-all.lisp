(in-package #:cl-acorn.benchmarks)

(defun run-all ()
  "Run all cl-acorn benchmarks and print results to *STANDARD-OUTPUT*.

Covers three sections:
  [AD Engine]       -- derivative and gradient computation at various dimensions
  [Distributions]   -- log-pdf evaluation for common distributions
  [Inference]       -- HMC, NUTS, and VI sampling throughput on 2D standard normal

Usage:
  (asdf:load-system :cl-acorn/benchmarks)
  (cl-acorn.benchmarks:run-all)"
  (format t "~%cl-acorn benchmark suite~%")
  (format t "========================~%")
  (format t "SBCL ~A  |  ~A~%"
          (lisp-implementation-version)
          (multiple-value-bind (s mi h d mo y)
              (decode-universal-time (get-universal-time))
            (declare (ignore s mi h))
            (format nil "~4,'0D-~2,'0D-~2,'0D" y mo d)))

  (print-bench-table "AD Engine" (run-ad-benchmarks))
  (print-bench-table "Distributions" (run-distributions-benchmarks))

  (let ((inf-results (run-inference-benchmarks)))
    (format t "~%[Inference]~%")
    (format t "~30A | ~12A |~%"
            "Task" "samples/sec")
    (format t "~30A-+-~12A-|~%"
            (make-string 30 :initial-element #\-)
            (make-string 12 :initial-element #\-))
    (dolist (r inf-results)
      (let ((sps (if (> (bench-result-mean-us r) 0.0d0)
                     (/ 1.0d6 (bench-result-mean-us r))
                     0.0d0)))
        (format t "~30A | ~12,1F |~%"
                (bench-result-name r)
                sps))))
  (values))
