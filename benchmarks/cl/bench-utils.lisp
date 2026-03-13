(in-package #:cl-acorn.benchmarks)

;;; bench-result struct

(defstruct bench-result
  "Timing result from a single benchmark task."
  (name       "" :type string)
  (mean-us  0.0d0 :type double-float)   ; mean microseconds per call
  (min-us   0.0d0 :type double-float)
  (max-us   0.0d0 :type double-float)
  (gc-bytes   0   :type integer))        ; bytes consed per call (approx)

;;; Low-level timer

(defun time-batch (thunk n-runs)
  "Run THUNK N-RUNS times and return mean microseconds per call."
  (let ((t0 (get-internal-real-time)))
    (dotimes (i n-runs)
      (declare (ignore i))
      (funcall thunk))
    (let ((elapsed (- (get-internal-real-time) t0)))
      (* 1.0d6
         (/ (float elapsed 0.0d0)
            (* (float n-runs 0.0d0)
               (float internal-time-units-per-second 0.0d0)))))))

;;; Main macro

(defmacro defbench (name (&key (n-runs 100) (n-trials 5) (n-warmup 10)) &body body)
  "Benchmark BODY, returning a BENCH-RESULT.
N-WARMUP discarded runs, then N-TRIALS batches of N-RUNS each.
Mean/min/max are computed across the trial batch-means."
  `(let ((thunk (lambda () ,@body)))
     ;; warmup
     (dotimes (i ,n-warmup)
       (declare (ignore i))
       (funcall thunk))
     ;; measure gc per run
     (let ((gc-start (sb-ext:get-bytes-consed)))
       (dotimes (i ,n-runs)
         (declare (ignore i))
         (funcall thunk))
       (let* ((gc-bytes-per-run
               (floor (- (sb-ext:get-bytes-consed) gc-start) ,n-runs))
              (trial-means
               (loop repeat ,n-trials
                     collect (time-batch thunk ,n-runs))))
         (make-bench-result
          :name ,name
          :mean-us (/ (reduce #'+ trial-means) (float ,n-trials 0.0d0))
          :min-us  (reduce #'min trial-means)
          :max-us  (reduce #'max trial-means)
          :gc-bytes gc-bytes-per-run)))))

;;; Table printers

(defun print-bench-table (section results)
  "Print RESULTS as a formatted table under SECTION heading."
  (format t "~%[~A]~%" section)
  (format t "~30A | ~9A | ~9A | ~9A | ~10A~%"
          "Task" "Mean(us)" "Min(us)" "Max(us)" "GC(bytes)")
  (format t "~30A-+-~9A-+-~9A-+-~9A-+-~10A~%"
          (make-string 30 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 10 :initial-element #\-))
  (dolist (r results)
    (format t "~30A | ~9,2F | ~9,2F | ~9,2F | ~10D~%"
            (bench-result-name r)
            (bench-result-mean-us r)
            (bench-result-min-us r)
            (bench-result-max-us r)
            (bench-result-gc-bytes r))))
