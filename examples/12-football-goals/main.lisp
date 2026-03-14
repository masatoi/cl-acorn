;;; examples/12-football-goals/main.lisp
;;;
;;; Workflow: AI-guided probabilistic model selection for football goal counts.
;;;
;;; An agent proposes three competing hypotheses about goal-scoring in
;;; international football, fits each with NUTS, then selects the best
;;; explanation of the data using WAIC and LOO.
;;;
;;; Hypotheses
;;; ----------
;;;   H1 — Baseline (2 params):
;;;     home_score_i ~ Poisson(lambda_h)
;;;     away_score_i ~ Poisson(lambda_a)
;;;     → Does separating home/away rates improve predictions?
;;;
;;;   H2 — Neutral adjustment (3 params):
;;;     non-neutral:  home_score_i ~ Poisson(lambda_h)
;;;                   away_score_i ~ Poisson(lambda_a)
;;;     neutral:      home_score_i ~ Poisson(lambda_n)
;;;                   away_score_i ~ Poisson(lambda_n)
;;;     → Does the venue type (home vs neutral ground) matter?
;;;
;;;   H3 — No home advantage (1 param):
;;;     home_score_i ~ Poisson(lambda)
;;;     away_score_i ~ Poisson(lambda)
;;;     → What if home advantage does not exist?
;;;
;;; All parameters are inferred on the log scale.
;;; Priors: Normal(0, 2) on each log-rate (weakly informative).
;;;
;;; Demonstrates:
;;;   - diag:run-chains + diag:print-convergence-summary  (per model)
;;;   - diag:waic / diag:loo / diag:print-model-comparison
;;;   - Programmatic ΔWAIC interpretation
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
;;; H1 — Baseline: separate home / away rates
;;; -------------------------------------------------------------------------

(defun make-log-posterior-h1 (data)
  "Log-posterior for H1.  Parameters: (log-lambda-h  log-lambda-a)"
  (lambda (params)
    (let* ((log-lh   (first params))
           (log-la   (second params))
           (lambda-h (ad:exp log-lh))
           (lambda-a (ad:exp log-la))
           (ll (reduce (lambda (acc row)
                         (ad:+ acc
                               (dist:poisson-log-pdf (first  row) :rate lambda-h)
                               (dist:poisson-log-pdf (second row) :rate lambda-a)))
                       data :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-lh :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-la :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-h1 (params row)
  "Point log-likelihood for H1 (plain arithmetic, no AD)."
  (let ((lh (exp (float (first  params) 0.0d0)))
        (la (exp (float (second params) 0.0d0))))
    (+ (dist:poisson-log-pdf (first  row) :rate lh)
       (dist:poisson-log-pdf (second row) :rate la))))

;;; -------------------------------------------------------------------------
;;; H2 — Neutral adjustment: neutral matches share a single rate
;;; -------------------------------------------------------------------------

(defun make-log-posterior-h2 (data)
  "Log-posterior for H2.
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
                         (let ((h (first row)) (a (second row)) (n (third row)))
                           (if n
                               (ad:+ acc
                                     (dist:poisson-log-pdf h :rate lambda-n)
                                     (dist:poisson-log-pdf a :rate lambda-n))
                               (ad:+ acc
                                     (dist:poisson-log-pdf h :rate lambda-h)
                                     (dist:poisson-log-pdf a :rate lambda-a)))))
                       data :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-lh :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-la :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-ln :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-h2 (params row)
  "Point log-likelihood for H2 (plain arithmetic, no AD)."
  (let ((lh (exp (float (first  params) 0.0d0)))
        (la (exp (float (second params) 0.0d0)))
        (ln (exp (float (third  params) 0.0d0))))
    (if (third row)
        (+ (dist:poisson-log-pdf (first  row) :rate ln)
           (dist:poisson-log-pdf (second row) :rate ln))
        (+ (dist:poisson-log-pdf (first  row) :rate lh)
           (dist:poisson-log-pdf (second row) :rate la)))))

;;; -------------------------------------------------------------------------
;;; H3 — No home advantage: single rate for all goals
;;; -------------------------------------------------------------------------

(defun make-log-posterior-h3 (data)
  "Log-posterior for H3.  Parameters: (log-lambda)"
  (lambda (params)
    (let* ((log-rate (first params))
           (rate     (ad:exp log-rate))
           (ll (reduce (lambda (acc row)
                         (ad:+ acc
                               (dist:poisson-log-pdf (first  row) :rate rate)
                               (dist:poisson-log-pdf (second row) :rate rate)))
                       data :initial-value 0.0d0)))
      (ad:+ ll (dist:normal-log-pdf log-rate :mu 0.0d0 :sigma 2.0d0)))))

(defun log-lik-h3 (params row)
  "Point log-likelihood for H3 (plain arithmetic, no AD)."
  (let ((rate (exp (float (first params) 0.0d0))))
    (+ (dist:poisson-log-pdf (first  row) :rate rate)
       (dist:poisson-log-pdf (second row) :rate rate))))

;;; -------------------------------------------------------------------------
;;; Posterior summary helpers
;;; -------------------------------------------------------------------------

(defun posterior-mean (samples param-index)
  "Compute the posterior mean of the PARAM-INDEX-th parameter across SAMPLES."
  (let ((n (length samples)))
    (/ (reduce #'+ samples :key (lambda (s) (nth param-index s)))
       (float n 0.0d0))))

(defun fit-model (name log-posterior-fn initial-params
                  &key (n-chains 4) (n-samples 1000) (n-warmup 500))
  "Fit a model, print convergence summary, and return the chain-result."
  (format t "  Fitting ~A...~%" name)
  (finish-output)
  (let ((result (diag:run-chains log-posterior-fn initial-params
                                 :n-chains  n-chains
                                 :n-samples n-samples
                                 :n-warmup  n-warmup)))
    (diag:print-convergence-summary result)
    result))

;;; -------------------------------------------------------------------------
;;; Step 4 helper: print interpretation given WAIC values and results
;;; -------------------------------------------------------------------------

(defun print-interpretation (waic-h1 waic-h2 waic-h3 result-h1 result-h2 result-h3)
  "Print a narrative interpretation of the model comparison results."
  (let* ((waics   (list (cons "H1-baseline" waic-h1)
                        (cons "H2-neutral"  waic-h2)
                        (cons "H3-no-adv"   waic-h3)))
         (best    (reduce (lambda (a b) (if (< (cdr a) (cdr b)) a b)) waics))
         (best-w  (cdr best))
         (all-s1  (apply #'append (diag:chain-result-samples result-h1)))
         (all-s2  (apply #'append (diag:chain-result-samples result-h2)))
         (all-s3  (apply #'append (diag:chain-result-samples result-h3))))

    (format t "[Step 4] Conclusion~%")
    (format t "~%  Best model: ~A  (WAIC = ~,1F)~%~%" (car best) best-w)

    (format t "  ΔWAIC relative to best model:~%")
    (dolist (entry waics)
      (format t "    ~12A  ~+,1F~%" (car entry) (- (cdr entry) best-w)))

    (format t "~%  Posterior means:~%")

    ;; H1 params
    (let ((lh1 (exp (posterior-mean all-s1 0)))
          (la1 (exp (posterior-mean all-s1 1))))
      (format t "    H1  lambda_home = ~,3F   lambda_away = ~,3F~%" lh1 la1))

    ;; H2 params
    (let ((lh2 (exp (posterior-mean all-s2 0)))
          (la2 (exp (posterior-mean all-s2 1)))
          (ln2 (exp (posterior-mean all-s2 2))))
      (format t "    H2  lambda_home = ~,3F   lambda_away = ~,3F   lambda_neutral = ~,3F~%"
              lh2 la2 ln2))

    ;; H3 param
    (let ((l3 (exp (posterior-mean all-s3 0))))
      (format t "    H3  lambda       = ~,3F~%" l3))

    (format t "~%  Interpretation:~%")

    ;; H1 vs H3: does home advantage exist?
    (let ((delta-h1-h3 (- waic-h3 waic-h1)))
      (if (> delta-h1-h3 10.0d0)
          (format t "    * H1 beats H3 by ΔWAIC = ~,1F: home advantage is real.~%"
                  delta-h1-h3)
          (format t "    * H1 and H3 are similar (ΔWAIC = ~,1F): ~
                       home advantage is weak.~%" delta-h1-h3)))

    ;; H2 vs H1: does neutral-ground distinction help?
    (let ((delta-h2-h1 (- waic-h1 waic-h2)))
      (if (> delta-h2-h1 10.0d0)
          (format t "    * H2 beats H1 by ΔWAIC = ~,1F: venue type (neutral vs home) ~
                       matters.~%" delta-h2-h1)
          (format t "    * H1 and H2 are similar (ΔWAIC = ~,1F): ~
                       neutral-ground distinction adds little.~%" (- delta-h2-h1))))))

;;; -------------------------------------------------------------------------
;;; Main workflow
;;; -------------------------------------------------------------------------

(defun main ()
  (let* ((data-path (merge-pathnames "examples/data/results.csv"
                                     (asdf:system-source-directory :cl-acorn)))
         (data      (load-goals data-path :from-year 2015 :to-year 2024))
         (n-neutral (count-if #'third data))
         (n-home    (- (length data) n-neutral)))

    ;; ------------------------------------------------------------------
    ;; Step 1: Propose hypotheses
    ;; ------------------------------------------------------------------
    (format t "~%=== AI-guided model selection: football goal counts (2015-2024) ===~%")
    (format t "Dataset: ~D matches  (~D home/away,  ~D neutral ground)~%~%"
            (length data) n-home n-neutral)

    (format t "[Step 1] Proposing hypotheses~%~%")
    (format t "  H1 (baseline):       home ~~ Pois(lambda_h), away ~~ Pois(lambda_a)~%")
    (format t "                       Question: does separating home/away rates help?~%~%")
    (format t "  H2 (neutral adj):    non-neutral as H1; neutral: both ~~ Pois(lambda_n)~%")
    (format t "                       Question: does the venue type matter?~%~%")
    (format t "  H3 (no home adv):    home = away ~~ Pois(lambda)~%")
    (format t "                       Question: what if home advantage does not exist?~%~%")

    ;; ------------------------------------------------------------------
    ;; Step 2: Fit each model
    ;; ------------------------------------------------------------------
    (format t "[Step 2] Fitting models via NUTS (4 chains x 1000 samples each)~%~%")

    (let ((result-h1 (fit-model "H1" (make-log-posterior-h1 data) '(0.4d0 0.18d0)))
          (result-h2 (fit-model "H2" (make-log-posterior-h2 data) '(0.4d0 0.18d0 0.3d0)))
          (result-h3 (fit-model "H3" (make-log-posterior-h3 data) '(0.3d0))))

      ;; ------------------------------------------------------------------
      ;; Step 3: Compare models
      ;; ------------------------------------------------------------------
      (format t "[Step 3] Model comparison (lower WAIC / LOO = better fit)~%~%")
      (diag:print-model-comparison
       "H1-baseline" result-h1 #'log-lik-h1 data
       "H2-neutral"  result-h2 #'log-lik-h2 data
       "H3-no-adv"   result-h3 #'log-lik-h3 data)
      (format t "~%")

      ;; ------------------------------------------------------------------
      ;; Step 4: Interpret results
      ;; ------------------------------------------------------------------
      (let ((waic-h1 (nth-value 0 (diag:waic result-h1 #'log-lik-h1 data)))
            (waic-h2 (nth-value 0 (diag:waic result-h2 #'log-lik-h2 data)))
            (waic-h3 (nth-value 0 (diag:waic result-h3 #'log-lik-h3 data))))
        (print-interpretation waic-h1 waic-h2 waic-h3
                              result-h1 result-h2 result-h3)))))

(main)
