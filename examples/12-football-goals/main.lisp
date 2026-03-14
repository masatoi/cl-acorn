;;; examples/12-football-goals/main.lisp
;;;
;;; Model comparison for international football goal counts.
;;;
;;; Three competing models are fitted and compared via WAIC and LOO:
;;;
;;;   Model A — Baseline (2 params):
;;;     home_score_i ~ Poisson(lambda_h)
;;;     away_score_i ~ Poisson(lambda_a)
;;;
;;;   Model B — Neutral adjustment (3 params):
;;;     non-neutral:  home_score_i ~ Poisson(lambda_h)
;;;                   away_score_i ~ Poisson(lambda_a)
;;;     neutral:      home_score_i ~ Poisson(lambda_n)
;;;                   away_score_i ~ Poisson(lambda_n)
;;;
;;;   Model C — No home advantage (1 param):
;;;     home_score_i ~ Poisson(lambda)
;;;     away_score_i ~ Poisson(lambda)
;;;
;;; All parameters are inferred on the log scale.
;;; Priors: Normal(0, 2) on each log-rate (weakly informative).
;;;
;;; Demonstrates:
;;;   - diag:run-chains    (multi-chain NUTS per model)
;;;   - diag:print-convergence-summary
;;;   - diag:waic / diag:loo / diag:print-model-comparison
;;;
;;; Data:
;;;   examples/data/results.csv
;;;   (martj42/international-football-results-from-1872-to-2017 on Kaggle)
;;;   CC0 1.0 licence
;;;
;;; Usage:
;;;   (load "examples/12-football-goals/main.lisp")

(asdf:load-system :cl-acorn)

;;; -------------------------------------------------------------------------
;;; Data loading
;;; -------------------------------------------------------------------------

(defun split-csv-line (line)
  "Split LINE on commas.  No quoted-field handling (not needed here)."
  (loop for start = 0 then (1+ pos)
        for pos = (position #\, line :start start)
        collect (subseq line start (or pos (length line)))
        while pos))

(defun load-goals (path &key (from-year 2015) (to-year 2024))
  "Return a list of (home-score away-score neutral-p) triples from PATH,
filtered to matches played in [FROM-YEAR, TO-YEAR].

PATH points to results.csv with columns:
  date, home_team, away_team, home_score, away_score,
  tournament, city, country, neutral"
  (with-open-file (stream path)
    (read-line stream nil)                     ; skip header
    (loop for line = (read-line stream nil)
          while line
          for fields = (split-csv-line line)
          for year = (when (= (length fields) 9)
                       (parse-integer (subseq (first fields) 0 4)
                                      :junk-allowed t))
          when (and year (>= year from-year) (<= year to-year))
          collect (list (parse-integer (fourth fields))
                        (parse-integer (fifth fields))
                        (string= (ninth fields) "TRUE")))))

;;; -------------------------------------------------------------------------
;;; Model A — Baseline: separate home / away rates, ignores neutral flag
;;; -------------------------------------------------------------------------

(defun make-log-posterior-a (data)
  "Log-posterior for Model A.
Parameters: (log-lambda-h  log-lambda-a)"
  (lambda (params)
    (let* ((log-lh   (first params))
           (log-la   (second params))
           (lambda-h (ad:exp log-lh))
           (lambda-a (ad:exp log-la))
           (ll (reduce (lambda (acc row)
                         (ad:+ acc
                               (dist:poisson-log-pdf (first  row) :rate lambda-h)
                               (dist:poisson-log-pdf (second row) :rate lambda-a)))
                       data
                       :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-lh :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-la :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-a (params row)
  "Point log-likelihood for Model A (plain arithmetic, no AD)."
  (let ((lh (exp (float (first  params) 0.0d0)))
        (la (exp (float (second params) 0.0d0))))
    (+ (dist:poisson-log-pdf (first  row) :rate lh)
       (dist:poisson-log-pdf (second row) :rate la))))

;;; -------------------------------------------------------------------------
;;; Model B — Neutral adjustment: neutral matches share a single rate
;;; -------------------------------------------------------------------------

(defun make-log-posterior-b (data)
  "Log-posterior for Model B.
Parameters: (log-lambda-h  log-lambda-a  log-lambda-n)
  lambda-h/a : home / away rate for non-neutral matches
  lambda-n   : common rate for both teams on neutral ground"
  (lambda (params)
    (let* ((log-lh   (first  params))
           (log-la   (second params))
           (log-ln   (third  params))
           (lambda-h (ad:exp log-lh))
           (lambda-a (ad:exp log-la))
           (lambda-n (ad:exp log-ln))
           (ll (reduce (lambda (acc row)
                         (let ((h-score  (first  row))
                               (a-score  (second row))
                               (neutralp (third  row)))
                           (if neutralp
                               (ad:+ acc
                                     (dist:poisson-log-pdf h-score :rate lambda-n)
                                     (dist:poisson-log-pdf a-score :rate lambda-n))
                               (ad:+ acc
                                     (dist:poisson-log-pdf h-score :rate lambda-h)
                                     (dist:poisson-log-pdf a-score :rate lambda-a)))))
                       data
                       :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-lh :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-la :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-ln :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-b (params row)
  "Point log-likelihood for Model B (plain arithmetic, no AD)."
  (let ((lh (exp (float (first  params) 0.0d0)))
        (la (exp (float (second params) 0.0d0)))
        (ln (exp (float (third  params) 0.0d0))))
    (if (third row)
        (+ (dist:poisson-log-pdf (first  row) :rate ln)
           (dist:poisson-log-pdf (second row) :rate ln))
        (+ (dist:poisson-log-pdf (first  row) :rate lh)
           (dist:poisson-log-pdf (second row) :rate la)))))

;;; -------------------------------------------------------------------------
;;; Model C — No home advantage: single rate for all goals
;;; -------------------------------------------------------------------------

(defun make-log-posterior-c (data)
  "Log-posterior for Model C.
Parameters: (log-lambda)
  All goals, home or away, follow the same Poisson rate."
  (lambda (params)
    (let* ((log-rate (first params))
           (rate     (ad:exp log-rate))
           (ll (reduce (lambda (acc row)
                         (ad:+ acc
                               (dist:poisson-log-pdf (first  row) :rate rate)
                               (dist:poisson-log-pdf (second row) :rate rate)))
                       data
                       :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-rate :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-c (params row)
  "Point log-likelihood for Model C (plain arithmetic, no AD)."
  (let ((rate (exp (float (first params) 0.0d0))))
    (+ (dist:poisson-log-pdf (first  row) :rate rate)
       (dist:poisson-log-pdf (second row) :rate rate))))

;;; -------------------------------------------------------------------------
;;; Posterior summary helpers
;;; -------------------------------------------------------------------------

(defun posterior-mean (samples param-index)
  "Compute the posterior mean of the PARAM-INDEX-th parameter."
  (let ((n (length samples)))
    (/ (reduce #'+ samples :key (lambda (s) (nth param-index s)))
       (float n 0.0d0))))

(defun print-posterior-means (result labels)
  "Print posterior means for each parameter in RESULT with LABELS."
  (let ((all-samples (apply #'append (diag:chain-result-samples result))))
    (format t "  Posterior means:~%")
    (loop for label in labels
          for i from 0
          for log-mean = (posterior-mean all-samples i)
          do (format t "    ~A = ~,4F  (log = ~,4F)~%"
                     label (exp log-mean) log-mean))))

;;; -------------------------------------------------------------------------
;;; Main
;;; -------------------------------------------------------------------------

(defun main ()
  (let* ((data-path (merge-pathnames "examples/data/results.csv"
                                     (asdf:system-source-directory :cl-acorn)))
         (data (load-goals data-path :from-year 2015 :to-year 2024))
         (n-neutral (count-if #'third data)))

    (format t "~%=== Football goals: model comparison (2015-2024) ===~%")
    (format t "Matches: ~D total  (~D neutral, ~D home/away)~%~%"
            (length data) n-neutral (- (length data) n-neutral))

    ;; ---- Model A ----
    (format t "--- Model A: baseline (lambda_h / lambda_a) ---~%")
    (let ((result-a (diag:run-chains (make-log-posterior-a data)
                                     '(0.4d0 0.18d0)
                                     :n-chains 4 :n-samples 1000 :n-warmup 500)))
      (diag:print-convergence-summary result-a)
      (print-posterior-means result-a '("lambda_home" "lambda_away"))
      (format t "~%")

      ;; ---- Model B ----
      (format t "--- Model B: neutral adjustment (+ lambda_n) ---~%")
      (let ((result-b (diag:run-chains (make-log-posterior-b data)
                                       '(0.4d0 0.18d0 0.3d0)
                                       :n-chains 4 :n-samples 1000 :n-warmup 500)))
        (diag:print-convergence-summary result-b)
        (print-posterior-means result-b '("lambda_home" "lambda_away" "lambda_neutral"))
        (format t "~%")

        ;; ---- Model C ----
        (format t "--- Model C: no home advantage (single lambda) ---~%")
        (let ((result-c (diag:run-chains (make-log-posterior-c data)
                                         '(0.3d0)
                                         :n-chains 4 :n-samples 1000 :n-warmup 500)))
          (diag:print-convergence-summary result-c)
          (print-posterior-means result-c '("lambda"))
          (format t "~%")

          ;; ---- Model comparison ----
          (format t "--- Model comparison (lower WAIC/LOO = better) ---~%")
          (diag:print-model-comparison
           "A-baseline"  result-a #'log-lik-a data
           "B-neutral"   result-b #'log-lik-b data
           "C-no-adv"    result-c #'log-lik-c data))))))

(main)
