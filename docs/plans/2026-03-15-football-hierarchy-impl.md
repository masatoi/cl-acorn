# Football Hierarchy Example Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `examples/13-football-hierarchy/main.lisp` — a single-file hierarchical Poisson
model that fits two models (flat H1 vs hierarchical H-hier) on international football data,
compares them with WAIC, and ranks 15 national teams by posterior attack/defense strength.

**Architecture:** Non-centered parameterization (35 parameters for N=15 teams) with soft
sum-to-zero identifiability constraint. Both models share the same filtered dataset
(15 named football nations, 2019-2024, non-neutral matches only). `diag:waic` and
`diag:print-model-comparison` handle model selection.

**Tech Stack:** cl-acorn public API only — `ad:`, `dist:`, `infer:`, `diag:` packages; standard CL.
Reference: `examples/12-football-goals/main.lisp` for structural patterns.

---

## Orientation

Before starting, read:
- `examples/12-football-goals/main.lisp` — structural pattern to follow (file header, 4-step narrative, `fit-model` helper, `diag:print-model-comparison` usage)
- `docs/plans/2026-03-15-football-hierarchy-design.md` — full model spec and output format

Key facts about cl-acorn's API:
- `dist:poisson-log-pdf k :rate r` — AD-transparent; `k` must be a non-negative integer value; `r` can be an AD value
- `dist:normal-log-pdf x :mu m :sigma s` — AD-transparent
- `ad:exp`, `ad:+`, `ad:-`, `ad:*`, `ad:/` — standard AD arithmetic
- `diag:run-chains log-pdf-fn initial-params :n-chains N :n-samples S :n-warmup W` — returns `chain-result`
- `diag:waic chain-result log-lik-fn data` → `(values waic p-waic lppd)`; `log-lik-fn` is `(lambda (params row) -> float)` using plain arithmetic (no AD)
- `diag:print-model-comparison "name-a" result-a log-lik-a data "name-b" result-b log-lik-b data ...`
- `diag:chain-result-samples result` → list of per-chain sample lists (each sample is a parameter list)
- `(apply #'append (diag:chain-result-samples result))` → flat list of all samples across chains
- **Format strings**: use `~~` for a literal tilde character (e.g., `"~~ Pois"` prints `~ Pois`)

---

## Task 1: Create file and data pipeline

**Files:**
- Create: `examples/13-football-hierarchy/main.lisp`

### Step 1: Create the file with header and data-loading functions

Write the complete file content below to `examples/13-football-hierarchy/main.lisp`:

```lisp
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
  (let ((team-index (build-team-index *focal-teams*))
        (team-set   (let ((ht (make-hash-table :test #'equal)))
                      (dolist (t *focal-teams*) (setf (gethash t ht) t))
                      ht)))
    (with-open-file (stream path)
      (read-line stream nil)                       ; skip header
      (loop for line = (read-line stream nil)
            while line
            for fields = (split-csv-line line)
            when (= (length fields) 9)
            for year    = (parse-integer (subseq (first fields) 0 4) :junk-allowed t)
            for home    = (second fields)
            for away    = (third  fields)
            for neutral = (ninth  fields)
            when (and year
                      (>= year from-year)
                      (<= year to-year)
                      (string= neutral "FALSE")
                      (gethash home team-set)
                      (gethash away team-set))
            collect (list (gethash home team-index)
                          (gethash away team-index)
                          (parse-integer (fourth fields))
                          (parse-integer (fifth  fields)))))))
```

### Step 2: Verify the data pipeline loads correctly

Load the file and check the output:

```lisp
(asdf:load-system :cl-acorn)
(load "examples/13-football-hierarchy/main.lisp")

;; Verify data loads
(let* ((data-path (merge-pathnames "examples/data/results.csv"
                                   (asdf:system-source-directory :cl-acorn)))
       (data (load-matches data-path)))
  (format t "Matches: ~D~%" (length data))
  (format t "First row: ~A~%" (first data))
  (format t "Team indices in range: ~A~%"
          (every (lambda (row)
                   (and (< (first row) 15) (< (second row) 15)
                        (>= (first row) 0) (>= (second row) 0)))
                 data)))
```

Expected output:
```
Matches: 179   (approximately — may vary by 1-5 due to CSV row count)
First row: (0 5 ...)   (Argentina=0, England=6, etc.)
Team indices in range: T
```

### Step 3: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add example 13: skeleton and data pipeline for football hierarchy model"
```

---

## Task 2: Flat baseline model (H1-flat)

**Files:**
- Modify: `examples/13-football-hierarchy/main.lisp` — append after `load-matches`

### Step 1: Add flat model functions after `load-matches`

Append this block to the file (after the data-loading section):

```lisp
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
```

### Step 2: Verify flat model log-posterior is finite

```lisp
(load "examples/13-football-hierarchy/main.lisp")

(let* ((data-path (merge-pathnames "examples/data/results.csv"
                                   (asdf:system-source-directory :cl-acorn)))
       (data (load-matches data-path))
       (lp   (make-log-posterior-flat data))
       (val  (funcall lp '(0.4d0 0.18d0))))
  (format t "log-posterior at initial params: ~A~%" val)
  (format t "finite? ~A~%" (and (realp val) (not (= val most-negative-double-float)))))
```

Expected: a negative finite float (e.g., `-612.3`), `finite? T`.

### Step 3: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add H1-flat log-posterior and log-likelihood for football hierarchy example"
```

---

## Task 3: Parameter decoder for hierarchical model

**Files:**
- Modify: `examples/13-football-hierarchy/main.lisp` — append after flat model section

### Step 1: Add decode-params

```lisp
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
```

### Step 2: Verify decode-params at REPL

```lisp
(load "examples/13-football-hierarchy/main.lisp")

;; Test with all-zero params (initial values)
(let* ((n 15)
       (params (make-list (+ 5 (* 2 n)) :initial-element 0.0d0)))
  (multiple-value-bind (home-adv mu-att sigma-att mu-def sigma-def attacks defenses)
      (decode-params params n)
    (format t "home-adv: ~A~%" home-adv)         ; 0.0
    (format t "sigma-att: ~A~%" sigma-att)        ; 1.0  (exp(0))
    (format t "attack-0: ~A~%" (first attacks))   ; 0.0  (0 + 1*0)
    (format t "n-attacks: ~A~%" (length attacks)) ; 15
    (format t "n-defenses: ~A~%" (length defenses)))) ; 15
```

Expected:
```
home-adv: 0.0d0
sigma-att: 1.0d0
attack-0: 0.0d0
n-attacks: 15
n-defenses: 15
```

### Step 3: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add decode-params for non-centered hierarchical parameterization"
```

---

## Task 4: Hierarchical log-posterior and log-likelihood

**Files:**
- Modify: `examples/13-football-hierarchy/main.lisp` — append after `decode-params`

### Step 1: Add make-log-posterior-hier and make-log-lik-hier

```lisp
(defun make-log-posterior-hier (data n-teams)
  "Log-posterior for H-hier.
Parameter vector layout: see decode-params.
Total parameters: 5 + 2*N-TEAMS = 35 for N-TEAMS=15."
  (let ((n (float n-teams 0.0d0)))
    (lambda (params)
      (multiple-value-bind (home-adv mu-att sigma-att mu-def sigma-def attacks defenses)
          (decode-params params n-teams)
        (declare (ignore mu-att sigma-att mu-def sigma-def))
        (let* (;; --- Likelihood ---
               (ll (reduce
                    (lambda (acc row)
                      (let* ((hi     (first  row))
                             (ai     (second row))
                             (hg     (third  row))
                             (ag     (fourth row))
                             (lrate-h (ad:+ home-adv
                                           (nth hi attacks)
                                           (ad:- (nth ai defenses))))
                             (lrate-a (ad:- (nth ai attacks)
                                           (nth hi defenses))))
                        (ad:+ acc
                              (dist:poisson-log-pdf hg :rate (ad:exp lrate-h))
                              (dist:poisson-log-pdf ag :rate (ad:exp lrate-a)))))
                    data :initial-value 0.0d0))
               ;; --- Priors on hyperparameters ---
               (lp (ad:+ (dist:normal-log-pdf (nth 0 params) :mu 0.0d0 :sigma 1.0d0)
                         (dist:normal-log-pdf (nth 1 params) :mu 0.0d0 :sigma 1.0d0)
                         (dist:normal-log-pdf (nth 2 params) :mu 0.0d0 :sigma 1.0d0)
                         (dist:normal-log-pdf (nth 3 params) :mu 0.0d0 :sigma 1.0d0)
                         (dist:normal-log-pdf (nth 4 params) :mu 0.0d0 :sigma 1.0d0)))
               ;; --- Priors on z offsets: z_i ~ Normal(0,1) ---
               (lz-att (reduce (lambda (acc z)
                                 (ad:+ acc (dist:normal-log-pdf z :mu 0.0d0 :sigma 1.0d0)))
                               (subseq params 5 (+ 5 n-teams))
                               :initial-value 0.0d0))
               (lz-def (reduce (lambda (acc z)
                                 (ad:+ acc (dist:normal-log-pdf z :mu 0.0d0 :sigma 1.0d0)))
                               (subseq params (+ 5 n-teams) (+ 5 (* 2 n-teams)))
                               :initial-value 0.0d0))
               ;; --- Soft sum-to-zero constraint: mean(z) ~ Normal(0, 0.01) ---
               (mean-z-att (ad:/ (reduce #'ad:+ (subseq params 5 (+ 5 n-teams))
                                         :initial-value 0.0d0)
                                 n))
               (mean-z-def (ad:/ (reduce #'ad:+ (subseq params (+ 5 n-teams)
                                                         (+ 5 (* 2 n-teams)))
                                         :initial-value 0.0d0)
                                 n))
               (lsz (ad:+ (dist:normal-log-pdf mean-z-att :mu 0.0d0 :sigma 0.01d0)
                          (dist:normal-log-pdf mean-z-def :mu 0.0d0 :sigma 0.01d0))))
          (ad:+ ll lp lz-att lz-def lsz))))))

(defun make-log-lik-hier (n-teams)
  "Return a point log-likelihood function for H-hier (plain arithmetic, no AD).
For use with diag:waic and diag:loo."
  (lambda (params row)
    (let* ((hi       (first  row))
           (ai       (second row))
           (hg       (third  row))
           (ag       (fourth row))
           (mu-att   (float (nth 1 params) 0.0d0))
           (s-att    (exp   (float (nth 2 params) 0.0d0)))
           (mu-def   (float (nth 3 params) 0.0d0))
           (s-def    (exp   (float (nth 4 params) 0.0d0)))
           (att-i    (+ mu-att (* s-att (float (nth (+ 5 hi) params) 0.0d0))))
           (def-i    (+ mu-def (* s-def (float (nth (+ 5 n-teams hi) params) 0.0d0))))
           (att-j    (+ mu-att (* s-att (float (nth (+ 5 ai) params) 0.0d0))))
           (def-j    (+ mu-def (* s-def (float (nth (+ 5 n-teams ai) params) 0.0d0))))
           (rate-h   (exp (+ (float (nth 0 params) 0.0d0) att-i (- def-j))))
           (rate-a   (exp (+ att-j (- def-i)))))
      (+ (dist:poisson-log-pdf hg :rate rate-h)
         (dist:poisson-log-pdf ag :rate rate-a)))))
```

### Step 2: Verify hierarchical log-posterior and gradient

```lisp
(load "examples/13-football-hierarchy/main.lisp")

(let* ((n 15)
       (data-path (merge-pathnames "examples/data/results.csv"
                                   (asdf:system-source-directory :cl-acorn)))
       (data    (load-matches data-path))
       (lp      (make-log-posterior-hier data n))
       (initial (make-list (+ 5 (* 2 n)) :initial-element 0.0d0))
       (val     (funcall lp initial)))
  (format t "log-posterior at zero: ~A~%" val)
  (format t "finite? ~A~%" (and (realp val) (> val most-negative-double-float))))

;; Check gradient has correct length
(let* ((n 15)
       (data-path (merge-pathnames "examples/data/results.csv"
                                   (asdf:system-source-directory :cl-acorn)))
       (data    (load-matches data-path))
       (lp      (make-log-posterior-hier data n))
       (initial (make-list (+ 5 (* 2 n)) :initial-element 0.0d0)))
  (multiple-value-bind (val grad) (ad:gradient lp initial)
    (format t "gradient length: ~A (expected ~A)~%" (length grad) (+ 5 (* 2 n)))
    (format t "all finite? ~A~%"
            (every (lambda (g) (and (realp g) (> g most-negative-double-float))) grad))))
```

Expected:
```
log-posterior at zero: -XXXX.XX  (large negative finite float)
finite? T
gradient length: 35 (expected 35)
all finite? T
```

### Step 3: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add hierarchical log-posterior and log-likelihood (non-centered, soft sum-to-zero)"
```

---

## Task 5: Posterior summary helpers

**Files:**
- Modify: `examples/13-football-hierarchy/main.lisp` — append after hierarchical model section

### Step 1: Add posterior summary functions

```lisp
;;; -------------------------------------------------------------------------
;;; Posterior summary helpers
;;; -------------------------------------------------------------------------

(defun posterior-mean-vec (samples idx)
  "Compute the posterior mean of the IDX-th parameter across all SAMPLES.
SAMPLES is a flat list of parameter vectors (one per post-warmup draw)."
  (/ (reduce #'+ samples :key (lambda (s) (float (nth idx s) 0.0d0)))
     (float (length samples) 0.0d0)))

(defun posterior-sd-vec (samples idx)
  "Compute the posterior standard deviation of the IDX-th parameter across SAMPLES."
  (let* ((n    (length samples))
         (mean (posterior-mean-vec samples idx))
         (ssq  (reduce #'+ samples
                       :key (lambda (s)
                              (let ((d (- (float (nth idx s) 0.0d0) mean)))
                                (* d d))))))
    (sqrt (/ ssq (float (max 1 (1- n)) 0.0d0)))))

(defun print-team-rankings (samples teams n-teams)
  "Print a ranked table of attack and defense posterior means +/- 1 SD.
SAMPLES: flat list of all post-warmup parameter vectors.
TEAMS:   list of team name strings (alphabetically sorted, matching parameter layout).
N-TEAMS: number of teams."
  ;; Compute attack_i = mu_att + sigma_att * z_att_i from raw samples
  ;; then derive per-team posterior mean and SD
  (let* ((attack-means
          (loop for i from 0 below n-teams
                collect (let ((att-i-samples
                               (mapcar (lambda (s)
                                         (let ((mu    (float (nth 1 s) 0.0d0))
                                               (sigma (exp (float (nth 2 s) 0.0d0)))
                                               (z     (float (nth (+ 5 i) s) 0.0d0)))
                                           (+ mu (* sigma z))))
                                       samples)))
                           (/ (reduce #'+ att-i-samples)
                              (float (length att-i-samples) 0.0d0)))))
         (attack-sds
          (loop for i from 0 below n-teams
                collect (let* ((att-i-samples
                                (mapcar (lambda (s)
                                          (let ((mu    (float (nth 1 s) 0.0d0))
                                                (sigma (exp (float (nth 2 s) 0.0d0)))
                                                (z     (float (nth (+ 5 i) s) 0.0d0)))
                                            (+ mu (* sigma z))))
                                        samples))
                               (n    (length att-i-samples))
                               (mean (/ (reduce #'+ att-i-samples) (float n 0.0d0)))
                               (ssq  (reduce #'+ att-i-samples
                                             :key (lambda (x) (let ((d (- x mean))) (* d d))))))
                           (sqrt (/ ssq (float (max 1 (1- n)) 0.0d0))))))
         (defense-means
          (loop for i from 0 below n-teams
                collect (let ((def-i-samples
                               (mapcar (lambda (s)
                                         (let ((mu    (float (nth 3 s) 0.0d0))
                                               (sigma (exp (float (nth 4 s) 0.0d0)))
                                               (z     (float (nth (+ 5 n-teams i) s) 0.0d0)))
                                           (+ mu (* sigma z))))
                                       samples)))
                           (/ (reduce #'+ def-i-samples)
                              (float (length def-i-samples) 0.0d0)))))
         (defense-sds
          (loop for i from 0 below n-teams
                collect (let* ((def-i-samples
                                (mapcar (lambda (s)
                                          (let ((mu    (float (nth 3 s) 0.0d0))
                                                (sigma (exp (float (nth 4 s) 0.0d0)))
                                                (z     (float (nth (+ 5 n-teams i) s) 0.0d0)))
                                            (+ mu (* sigma z))))
                                        samples))
                               (n    (length def-i-samples))
                               (mean (/ (reduce #'+ def-i-samples) (float n 0.0d0)))
                               (ssq  (reduce #'+ def-i-samples
                                             :key (lambda (x) (let ((d (- x mean))) (* d d))))))
                           (sqrt (/ ssq (float (max 1 (1- n)) 0.0d0))))))
         ;; Sort by attack mean descending
         (ranking (sort (loop for i from 0 below n-teams
                              collect (list i
                                            (nth i attack-means)
                                            (nth i attack-sds)
                                            (nth i defense-means)
                                            (nth i defense-sds)))
                        #'> :key #'second)))
    (format t "~%  ~4A  ~-14A  ~7A  ~5A  ~8A  ~5A~%"
            "Rank" "Team" "Attack" "±SD" "Defense" "±SD")
    (format t "  ~56A~%" (make-string 56 :initial-element #\=))
    (loop for (i att att-sd def def-sd) in ranking
          for rank from 1
          do (format t "  ~4D  ~-14A  ~+7,3F  ~5,3F  ~+8,3F  ~5,3F~%"
                     rank (nth i teams) att att-sd def def-sd))
    (format t "  ~56A~%" (make-string 56 :initial-element #\=))
    (format t "  (attack > 0: scores more than avg; defense < 0: concedes less)~%")))
```

### Step 2: Verify posterior-mean-vec at REPL with synthetic data

```lisp
(load "examples/13-football-hierarchy/main.lisp")

;; Synthetic: 10 samples, all zeros except idx 0 = 1.0
(let ((fake-samples (loop repeat 10 collect (make-list 35 :initial-element 0.0d0))))
  ;; Set idx 0 = 1.0 in each sample
  (setf fake-samples
        (mapcar (lambda (s) (cons 1.0d0 (rest s))) fake-samples))
  (format t "mean of idx 0: ~A (expected 1.0)~%" (posterior-mean-vec fake-samples 0))
  (format t "mean of idx 1: ~A (expected 0.0)~%" (posterior-mean-vec fake-samples 1))
  (format t "sd   of idx 0: ~A (expected 0.0)~%" (posterior-sd-vec   fake-samples 0)))
```

Expected:
```
mean of idx 0: 1.0d0 (expected 1.0)
mean of idx 1: 0.0d0 (expected 0.0)
sd   of idx 0: 0.0d0 (expected 0.0)
```

### Step 3: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add posterior summary helpers: posterior-mean-vec, posterior-sd-vec, print-team-rankings"
```

---

## Task 6: Fit helper and main 4-step workflow

**Files:**
- Modify: `examples/13-football-hierarchy/main.lisp` — append the `fit-model`, `print-hier-interpretation`, and `main` functions, then add `(main)` call at the end

### Step 1: Add fit-model, print-hier-interpretation, main, and (main) call

```lisp
;;; -------------------------------------------------------------------------
;;; Fit helper
;;; -------------------------------------------------------------------------

(defun fit-model (name log-posterior-fn initial-params
                  &key (n-chains 2) (n-samples 500) (n-warmup 300))
  "Fit a model with NUTS, print convergence summary, and return the chain-result."
  (format t "  Fitting ~A...~%" name)
  (finish-output)
  (let ((result (diag:run-chains log-posterior-fn initial-params
                                 :n-chains  n-chains
                                 :n-samples n-samples
                                 :n-warmup  n-warmup)))
    (diag:print-convergence-summary result)
    result))

;;; -------------------------------------------------------------------------
;;; Step 4 helper: interpret WAIC comparison and print team rankings
;;; -------------------------------------------------------------------------

(defun print-hier-interpretation (waic-flat waic-hier result-hier n-teams)
  "Print ΔWAIC interpretation, hyperparameter estimates, and team ranking table."
  (let* ((all-samples (apply #'append (diag:chain-result-samples result-hier)))
         (delta-waic  (- waic-flat waic-hier))
         (eta-mean    (posterior-mean-vec all-samples 0))
         (eta-sd      (posterior-sd-vec   all-samples 0))
         (s-att-mean  (exp (posterior-mean-vec all-samples 2)))
         (s-def-mean  (exp (posterior-mean-vec all-samples 4))))
    (format t "[Step 4] Interpretation~%~%")
    (format t "  ΔWAIC (flat -> hier) = ~+,1F~%" delta-waic)
    (if (> delta-waic 10.0d0)
        (format t "  -> Hierarchical model improves predictive accuracy.~%~%")
        (format t "  -> Models are comparable (ΔWAIC < 10).~%~%"))
    (format t "  Hyperparameter estimates:~%")
    (format t "    Home advantage  eta    = ~+,3F  +/- ~,3F~%" eta-mean eta-sd)
    (format t "    Attack spread   sigma  = ~,3F  (heterogeneity across teams)~%" s-att-mean)
    (format t "    Defense spread  sigma  = ~,3F~%~%" s-def-mean)
    (format t "  Team strength rankings (posterior mean +/- 1 SD):~%")
    (print-team-rankings all-samples *focal-teams* n-teams)
    (format t "~%  Partial pooling:~%")
    (let* ((match-counts
            (let ((cnt (make-array n-teams :initial-element 0)))
              (dolist (row (let ((data-path
                                  (merge-pathnames "examples/data/results.csv"
                                                   (asdf:system-source-directory :cl-acorn))))
                             (load-matches data-path)))
                (incf (aref cnt (first  row)))
                (incf (aref cnt (second row))))
              cnt))
           (sorted-teams (sort (loop for i from 0 below n-teams
                                     collect (cons (aref match-counts i) (nth i *focal-teams*)))
                               #'>  :key #'car))
           (most-team  (cdr (first  sorted-teams)))
           (least-team (cdr (car (last sorted-teams))))
           (most-idx   (position most-team  *focal-teams* :test #'string=))
           (least-idx  (position least-team *focal-teams* :test #'string=))
           ;; SD of attack for each team
           (most-att-sd  (let* ((samps (mapcar (lambda (s)
                                                 (let ((mu (float (nth 1 s) 0.0d0))
                                                       (sg (exp (float (nth 2 s) 0.0d0)))
                                                       (z  (float (nth (+ 5 most-idx) s) 0.0d0)))
                                                   (+ mu (* sg z))))
                                               all-samples))
                                (n (length samps))
                                (m (/ (reduce #'+ samps) (float n 0.0d0)))
                                (v (/ (reduce #'+ samps :key (lambda (x) (let ((d (- x m))) (* d d))))
                                      (float (max 1 (1- n)) 0.0d0))))
                           (sqrt v)))
           (least-att-sd (let* ((samps (mapcar (lambda (s)
                                                 (let ((mu (float (nth 1 s) 0.0d0))
                                                       (sg (exp (float (nth 2 s) 0.0d0)))
                                                       (z  (float (nth (+ 5 least-idx) s) 0.0d0)))
                                                   (+ mu (* sg z))))
                                               all-samples))
                                (n (length samps))
                                (m (/ (reduce #'+ samps) (float n 0.0d0)))
                                (v (/ (reduce #'+ samps :key (lambda (x) (let ((d (- x m))) (* d d))))
                                      (float (max 1 (1- n)) 0.0d0))))
                           (sqrt v))))
      (format t "    ~A  (~D matches): attack SD = ~,3F~%"
              most-team (aref match-counts most-idx) most-att-sd)
      (format t "    ~A (~D matches): attack SD = ~,3F~%"
              least-team (aref match-counts least-idx) least-att-sd)
      (format t "    Fewer matches -> wider credible intervals (partial pooling in action).~%"))))

;;; -------------------------------------------------------------------------
;;; Main workflow
;;; -------------------------------------------------------------------------

(defun main ()
  (let* ((data-path (merge-pathnames "examples/data/results.csv"
                                     (asdf:system-source-directory :cl-acorn)))
         (data      (load-matches data-path))
         (n-teams   (length *focal-teams*))
         (n-matches (length data)))

    ;; ----------------------------------------------------------------
    ;; Step 1: Propose models
    ;; ----------------------------------------------------------------
    (format t "~%=== AI-guided model selection: team strengths in international football ===~%")
    (format t "Dataset: ~D matches  (~D teams, 2019-2024, home/away only)~%~%"
            n-matches n-teams)

    (format t "[Step 1] Proposing models~%~%")
    (format t "  H1-flat (baseline, 2 params):~%")
    (format t "    home_goals ~~ Pois(exp(log_lh)),  away_goals ~~ Pois(exp(log_la))~%")
    (format t "    Question: can a single home/away rate explain all matches?~%~%")
    (format t "  H-hier  (hierarchical, ~D params):~%" (+ 5 (* 2 n-teams)))
    (format t "    home_goals ~~ Pois(exp(eta + att_i - def_j))~%")
    (format t "    away_goals ~~ Pois(exp(att_j - def_i))~%")
    (format t "    att_i = mu_att + sigma_att * z_att_i  (non-centered)~%")
    (format t "    Question: does team identity improve predictive accuracy?~%~%")

    ;; ----------------------------------------------------------------
    ;; Step 2: Fit both models
    ;; ----------------------------------------------------------------
    (format t "[Step 2] Fitting models via NUTS (2 chains x 500 samples each)~%~%")

    (let* ((initial-flat  '(0.4d0 0.18d0))
           (initial-hier  (make-list (+ 5 (* 2 n-teams)) :initial-element 0.0d0))
           (result-flat  (fit-model "H1-flat"
                                    (make-log-posterior-flat data)
                                    initial-flat))
           (result-hier  (fit-model "H-hier"
                                    (make-log-posterior-hier data n-teams)
                                    initial-hier))
           (log-lik-flat-fn (make-log-lik-flat))
           (log-lik-hier-fn (make-log-lik-hier n-teams)))

      ;; ----------------------------------------------------------------
      ;; Step 3: Compare models
      ;; ----------------------------------------------------------------
      (format t "[Step 3] Model comparison (lower WAIC / LOO = better fit)~%~%")
      (diag:print-model-comparison
       "H1-flat"  result-flat  log-lik-flat-fn data
       "H-hier"   result-hier  log-lik-hier-fn data)
      (format t "~%")

      ;; ----------------------------------------------------------------
      ;; Step 4: Interpret
      ;; ----------------------------------------------------------------
      (let ((waic-flat (nth-value 0 (diag:waic result-flat  log-lik-flat-fn data)))
            (waic-hier (nth-value 0 (diag:waic result-hier  log-lik-hier-fn data))))
        (print-hier-interpretation waic-flat waic-hier result-hier n-teams)))))

(main)
```

### Step 2: Run a fast smoke test before full run

```lisp
;; Smoke test: 1 chain, 50 samples — just checks nothing crashes
(asdf:load-system :cl-acorn)
(load "examples/13-football-hierarchy/main.lisp")
```

If the smoke test runs without error, proceed to the full run.

### Step 3: Run the full example

```bash
sbcl --non-interactive --load examples/13-football-hierarchy/main.lisp
```

Expected output structure (values will vary):
```
=== AI-guided model selection: team strengths in international football ===
Dataset: 179 matches  (15 teams, 2019-2024, home/away only)

[Step 1] Proposing models
  ...

[Step 2] Fitting models via NUTS (2 chains x 500 samples each)
  Fitting H1-flat...
  Convergence diagnostics  (2 chains x 500 samples)
  ...
  Fitting H-hier...
  Convergence diagnostics  (2 chains x 500 samples)
  ...

[Step 3] Model comparison
  =====================================================
  Model    | WAIC    | p_waic | LOO    | p_loo
  =====================================================
  H1-flat  | XXXX.X  |    1.X | XXXX.X |   1.X
  H-hier   | XXXX.X  |   XX.X | XXXX.X |  XX.X
  =====================================================

[Step 4] Interpretation
  ΔWAIC (flat -> hier) = +XX.X
  ...
  Team strength rankings:
  ...
```

Check that:
- R-hat < 1.1 for all parameters (or at least most z_i values)
- WAIC values are finite negative numbers
- Team rankings table prints with 15 rows

### Step 4: Commit

```bash
git add examples/13-football-hierarchy/main.lisp
git commit -m "Add fit-model, main workflow, and 4-step narrative for football hierarchy example"
```

---

## Task 7: Update README and final smoke test

**Files:**
- Modify: `README.md` — add example 13 to the examples table

### Step 1: Find the example 12 row in README.md

Look at `README.md` around line 351 (the examples table). Find:
```markdown
| [12-football-goals](examples/12-football-goals/) | Baseline Poisson model for international football goal counts with multi-chain NUTS and convergence diagnostics |
```

### Step 2: Add example 13 row immediately after example 12

Insert after the example 12 row:
```markdown
| [13-football-hierarchy](examples/13-football-hierarchy/) | Hierarchical Poisson model for national team strengths with non-centered parameterization, partial pooling, and flat vs hierarchical WAIC comparison |
```

Also update the test count in README.md line:
```markdown
cl-acorn uses [Rove](https://github.com/fukamachi/rove) for testing (currently 207 tests).
```
This line documents the test suite count — the examples don't add tests, so this stays at 207.

### Step 3: Run full test suite to verify nothing is broken

```bash
sbcl --non-interactive \
  --eval "(require 'asdf)" \
  --eval "(push #p\"/home/wiz/cl-acorn/\" asdf:*central-registry*)" \
  --eval "(asdf:test-system :cl-acorn)" \
  --eval "(sb-ext:exit)"
```

Expected: all 207 tests pass.

### Step 4: Commit

```bash
git add README.md
git commit -m "Update README: add example 13 football hierarchy to examples table"
```

---

## Troubleshooting

### R-hat > 1.1 for z_i parameters
Increase warmup or samples:
```lisp
(fit-model "H-hier" lp initial :n-chains 2 :n-samples 1000 :n-warmup 500)
```
The soft sum-to-zero constraint (`sigma=0.01`) may need tightening if mean(z) drifts — try `0.001`.

### `dist:poisson-log-pdf` type error
The function requires `k` to be an integer value. Data from `parse-integer` returns fixnums, which satisfy the check. If you see a type error, verify that `(integerp (third row))` is `T` for your data rows.

### `subseq` on parameter list — unexpected behavior
`subseq` works on lists in Common Lisp and returns a fresh list. If the parameter vector is a vector (not a list), `(nth ...)` still works but `subseq` will also work on vectors. `diag:run-chains` and `infer:nuts` work with lists internally, so `params` will always be a list.

### Divergences in H-hier
Some divergences are expected in a 35-dimensional hierarchical model. If divergence rate > 10%:
1. The non-centered parameterization already addresses Neal's funnel
2. Try reducing `step-size` in `diag:run-chains`: `:step-size 0.05d0`
3. The soft sum-to-zero constraint with `sigma=0.01` is strong — if divergences cluster near the constraint, try `sigma=0.1`

### Negative p_loo values
This is a known limitation of the current `diag:loo` implementation (documented in the test suite). It does not affect model ranking — use WAIC as the primary criterion.
