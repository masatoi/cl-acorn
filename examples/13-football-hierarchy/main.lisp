;;; examples/13-football-hierarchy/main.lisp
;;;
;;; Workflow: AI-guided hierarchical model for international football team strengths.
;;;
;;; An agent proposes two competing models and selects the better predictor
;;; of goal counts using WAIC, while estimating team-specific attack and defense
;;; strengths via partial pooling.
;;;
;;; Models
;;; ------
;;;   H1-flat (baseline, 2 params):
;;;     home_goals_ij ~ Poisson(exp(log_lambda_h))
;;;     away_goals_ij ~ Poisson(exp(log_lambda_a))
;;;     → Global home/away rates; team identity ignored.
;;;
;;;   H-hier (hierarchical, 5 + 2*N params, N=15 teams):
;;;     home_goals_ij ~ Poisson(exp(home_adv + attack_i - defense_j))
;;;     away_goals_ij ~ Poisson(exp(attack_j - defense_i))
;;;     attack_i  = mu_att + sigma_att * z_att_i
;;;     defense_i = mu_def + sigma_def * z_def_i
;;;     → Non-centered parameterization; partial pooling over teams.
;;;
;;; Demonstrates:
;;;   - Hierarchical models with non-centered parameterization
;;;   - Partial pooling: teams with few matches borrow strength from the population
;;;   - Soft sum-to-zero identifiability constraint
;;;   - Flat vs hierarchical model comparison with WAIC
;;;   - Team strength ranking from posterior samples
;;;
;;; Data:
;;;   examples/data/results.csv
;;;   (martj42/international-football-results-from-1872-to-2017 on Kaggle)
;;;   CC0 1.0 licence
;;;
;;; Usage:
;;;   (load "examples/13-football-hierarchy/main.lisp")

(asdf:load-system :cl-acorn)

;;; -------------------------------------------------------------------------
;;; Team selection and data loading
;;; -------------------------------------------------------------------------

(defvar *focal-teams*
  '("Argentina" "Belgium" "Brazil" "Colombia" "Croatia"
    "Denmark" "England" "France" "Germany" "Italy"
    "Netherlands" "Portugal" "Spain" "Switzerland" "Uruguay")
  "Alphabetically sorted list of 15 focal national teams.")

(defun build-team-index (teams)
  "Return a hash-table mapping each team name to its 0-based index in TEAMS.
TEAMS must already be sorted (alphabetical order gives stable indices)."
  (let ((ht (make-hash-table :test #'equal)))
    (loop for team in teams
          for idx from 0
          do (setf (gethash team ht) idx))
    ht))

(defun split-csv-line (line)
  "Split LINE on commas. No quoted-field handling (not needed here)."
  (loop for start = 0 then (1+ pos)
        for pos = (position #\, line :start start)
        collect (subseq line start (or pos (length line)))
        while pos))

(defun load-matches (path &key (from-year 2019) (to-year 2024))
  "Return a list of (home-team-idx away-team-idx home-goals away-goals) 4-tuples
from PATH, filtered to non-neutral matches in [FROM-YEAR, TO-YEAR] where both
teams appear in *FOCAL-TEAMS*.

PATH points to results.csv with columns:
  date, home_team, away_team, home_score, away_score,
  tournament, city, country, neutral"
  (let ((team-index (build-team-index *focal-teams*)))
    (with-open-file (stream path)
      (read-line stream nil)                       ; skip header
      (loop for line = (read-line stream nil)
            while line
            for fields = (split-csv-line line)
            for year    = (when (= (length fields) 9)
                            (parse-integer (subseq (first fields) 0 4)
                                           :junk-allowed t))
            for home    = (when (= (length fields) 9) (second fields))
            for away    = (when (= (length fields) 9) (third  fields))
            for neutral = (when (= (length fields) 9) (ninth  fields))
            when (and year
                      (>= year from-year)
                      (<= year to-year)
                      (string= neutral "FALSE")
                      (gethash home team-index)
                      (gethash away team-index))
            collect (list (gethash home team-index)
                          (gethash away team-index)
                          (parse-integer (fourth fields))
                          (parse-integer (fifth  fields)))))))

;;; -------------------------------------------------------------------------
;;; H1-flat — Baseline: global home/away rates (no team identity)
;;; -------------------------------------------------------------------------

(defun make-log-posterior-flat (data)
  "Log-posterior for H1-flat.  Parameters: (log-lambda-h  log-lambda-a)"
  (lambda (params)
    (let* ((log-lh   (first  params))
           (log-la   (second params))
           (lambda-h (ad:exp log-lh))
           (lambda-a (ad:exp log-la))
           (ll (reduce (lambda (acc row)
                         (ad:+ acc
                               (dist:poisson-log-pdf (third  row) :rate lambda-h)
                               (dist:poisson-log-pdf (fourth row) :rate lambda-a)))
                       data :initial-value 0.0d0)))
      (ad:+ ll
            (dist:normal-log-pdf log-lh :mu 0.0d0 :sigma 2.0d0)
            (dist:normal-log-pdf log-la :mu 0.0d0 :sigma 2.0d0)))))

(defun make-log-lik-flat ()
  "Return a point log-likelihood function for H1-flat (plain arithmetic, no AD).
For use with diag:waic and diag:loo."
  (lambda (params row)
    (let ((lh (exp (float (first  params) 0.0d0)))
          (la (exp (float (second params) 0.0d0))))
      (+ (dist:poisson-log-pdf (third  row) :rate lh)
         (dist:poisson-log-pdf (fourth row) :rate la)))))

;;; -------------------------------------------------------------------------
;;; H-hier — Hierarchical: non-centered parameterization
;;; -------------------------------------------------------------------------

(defun decode-params (params n-teams)
  "Destructure flat parameter vector PARAMS into named hierarchical components.
Returns (values home-adv mu-att sigma-att mu-def sigma-def attacks defenses).
ATTACKS and DEFENSES are lists of N-TEAMS AD values:
  attack_i  = mu-att + sigma-att * z_att_i
  defense_i = mu-def + sigma-def * z_def_i

Parameter layout (length = 5 + 2*N-TEAMS):
  index 0         : home-adv
  index 1         : mu-att
  index 2         : log-sigma-att   (sigma-att = exp of this)
  index 3         : mu-def
  index 4         : log-sigma-def
  indices 5..N+4  : z-att-0 .. z-att-(N-1)
  indices N+5..end: z-def-0 .. z-def-(N-1)"
  (let* ((home-adv   (nth 0 params))
         (mu-att     (nth 1 params))
         (sigma-att  (ad:exp (nth 2 params)))
         (mu-def     (nth 3 params))
         (sigma-def  (ad:exp (nth 4 params)))
         (z-att      (subseq params 5 (+ 5 n-teams)))
         (z-def      (subseq params (+ 5 n-teams) (+ 5 (* 2 n-teams))))
         (attacks    (mapcar (lambda (z) (ad:+ mu-att (ad:* sigma-att z))) z-att))
         (defenses   (mapcar (lambda (z) (ad:+ mu-def (ad:* sigma-def z))) z-def)))
    (values home-adv mu-att sigma-att mu-def sigma-def attacks defenses)))
