# Football Hierarchy Example Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan from this design doc.

**Goal:** Create `examples/13-football-hierarchy/main.lisp` — a hierarchical Poisson model for
international football that demonstrates partial pooling of team-specific attack and defense
strengths, comparing against the flat baseline (example 12 H1) via WAIC.

**Architecture:** Two models are fitted and compared. The flat H1 (2 parameters) is re-implemented
on the same filtered dataset. The hierarchical H-hier (5 + 2N parameters, N=15 teams) uses
non-centered parameterization with a soft sum-to-zero identifiability constraint. Both are evaluated
with `diag:waic` and `diag:loo`.

**Tech Stack:** cl-acorn public API only — `ad:`, `dist:`, `infer:`, `diag:` packages; standard CL.

---

## 1. Data Pipeline

### Team Selection

Fixed list of 15 recognizable football nations:

```lisp
(defvar *focal-teams*
  '("Brazil" "Argentina" "France" "Germany" "Spain"
    "England" "Italy" "Netherlands" "Portugal" "Belgium"
    "Uruguay" "Croatia" "Denmark" "Switzerland" "Colombia"))
```

This list is sorted alphabetically at load time to build a stable index map
(`team-name → 0-based integer index`).

### Data Loading

Source: `examples/data/results.csv`
Fields: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`

Filtering rules:
1. Year in `[from-year, to-year]` (default 2019–2024)
2. Both `home_team` and `away_team` are members of `*focal-teams*`
3. `neutral = FALSE` (home-advantage matches only)

Output: list of 4-tuples `(home-team-idx away-team-idx home-goals away-goals)`
where indices are 0-based positions in the sorted team list.

Expected data volume: ~180 matches, 15 teams, ~12–31 matches per team.

---

## 2. Model Specification

### Flat baseline H1 (re-implemented on filtered data)

```
home_goals_ij ~ Poisson(exp(log_λ_h))
away_goals_ij ~ Poisson(exp(log_λ_a))

Priors:
  log_λ_h ~ Normal(0, 2)
  log_λ_a ~ Normal(0, 2)

theta: [log_λ_h, log_λ_a]   (2 parameters)
```

This is identical to example 12 H1 but fitted only on the 15-team filtered dataset.
The `log-lik-flat` function is reused for `diag:waic`.

### Hierarchical model H-hier

```
home_goals_ij ~ Poisson(exp(home_adv + attack_i - defense_j))
away_goals_ij ~ Poisson(exp(attack_j - defense_i))
```

#### Non-centered parameterization

```
theta: [home_adv,                    ; index 0
        mu_att, log_sigma_att,       ; indices 1-2
        mu_def, log_sigma_def,       ; indices 3-4
        z_att_0 .. z_att_{N-1},      ; indices 5 .. N+4
        z_def_0 .. z_def_{N-1}]      ; indices N+5 .. 2N+4

Derived quantities (computed inside log-posterior):
  sigma_att  = exp(log_sigma_att)
  sigma_def  = exp(log_sigma_def)
  attack_i   = mu_att + sigma_att * z_att_i
  defense_i  = mu_def + sigma_def * z_def_i
```

Total parameters: `5 + 2N = 35` for N=15.

#### Priors

```
home_adv        ~ Normal(0, 1)
mu_att          ~ Normal(0, 1)
mu_def          ~ Normal(0, 1)
log_sigma_att   ~ Normal(0, 1)   ; → sigma_att = exp(.) ∈ (0, ∞)
log_sigma_def   ~ Normal(0, 1)
z_att_i         ~ Normal(0, 1)   ; for i = 0 .. N-1
z_def_i         ~ Normal(0, 1)   ; for i = 0 .. N-1
```

#### Soft sum-to-zero identifiability constraint

```
mean(z_att) ~ Normal(0, 0.01)
mean(z_def) ~ Normal(0, 0.01)
```

Added as two extra `dist:normal-log-pdf` terms in the log-posterior.
`sigma = 0.01` makes this effectively a hard constraint while keeping gradients smooth.

#### Initial parameter vector

```lisp
(make-list (+ 5 (* 2 n-teams)) :initial-element 0.0d0)
```

At `theta = 0`: `attack_i = 0`, `defense_i = 0`, `home_adv = 0`, `rate = exp(0) = 1.0`.
NUTS warmup adapts from there.

---

## 3. Implementation Structure

### Key functions

| Function | Purpose |
|----------|---------|
| `build-team-index (teams)` | Returns sorted list + hash-table `name→idx` |
| `load-matches (path &key from-year to-year teams)` | Returns list of 4-tuples |
| `make-log-posterior-flat (data)` | Closure for H1 flat model |
| `log-lik-flat (params row)` | Point log-likelihood for H1 (plain arithmetic) |
| `decode-params (params n-teams)` | Destructures theta → named components (returns multiple values) |
| `make-log-posterior-hier (data n-teams)` | Closure for H-hier using `decode-params` |
| `log-lik-hier (params row n-teams)` | Point log-likelihood for H-hier (plain arithmetic) |
| `posterior-mean-vec (samples idx)` | Mean of parameter at index `idx` across all chain samples |
| `posterior-sd-vec (samples idx)` | SD of parameter at index `idx` |
| `print-team-rankings (samples teams n-teams)` | Formatted attack/defense ranking table |
| `main ()` | 4-step workflow |

### decode-params detail

```lisp
(defun decode-params (params n-teams)
  ;; Returns (values home-adv mu-att sigma-att mu-def sigma-def attacks defenses)
  ;; attacks, defenses: lists of length n-teams, each element an AD value
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

---

## 4. Output Format

### [Step 1] Hypothesis proposal (printed)

```
=== AI-guided model selection: team strengths in international football ===
Dataset: NNN matches  (15 teams, 2019-2024, home/away only)

[Step 1] Proposing models

  H1-flat  (baseline): home ~ Pois(exp(λ_h)),  away ~ Pois(exp(λ_a))
                        2 parameters — global home/away rates, no team identity

  H-hier   (hierarchical): home ~ Pois(exp(η + att_i - def_j))
                            away ~ Pois(exp(att_j - def_i))
                            35 parameters — partial pooling over 15 teams
```

### [Step 2] Fitting (printed via diag:print-convergence-summary)

Runtime target: < 3 minutes total.
- H1-flat:  2 chains × 500 samples (fast, 2 params)
- H-hier:  2 chains × 500 samples (35 params; increase if R-hat > 1.05)

Convergence summary for H-hier prints R-hat/ESS grouped by parameter type
(hyperparams, z_att, z_def) rather than 35 individual rows.

### [Step 3] Model comparison table (via diag:print-model-comparison)

```
[Step 3] Model comparison

  =====================================================
  Model    | WAIC    | p_waic | LOO    | p_loo
  =====================================================
  H1-flat  | XXXX.X  |    1.8 | XXXX.X |   1.9
  H-hier   | XXXX.X  |   18.4 | XXXX.X |  19.1
  =====================================================
  Lower is better.
```

### [Step 4] Interpretation

```
[Step 4] Interpretation

  ΔWAIC (flat → hier) = +XX.X
  → [hierarchical model improves predictions / models are comparable]

  Home advantage:  η = X.XX ± X.XX
  Attack spread:   σ_att = X.XX  (heterogeneity across teams)
  Defense spread:  σ_def = X.XX

  Team strength rankings (posterior mean ± 1 SD):
  =====================================================
  Rank  Team           Attack   ±SD   Defense   ±SD
  =====================================================
     1  France          +X.XX  X.XX    -X.XX   X.XX
     2  Brazil          +X.XX  X.XX    -X.XX   X.XX
   ...
    15  Colombia        -X.XX  X.XX    +X.XX   X.XX
  =====================================================
  (attack > 0 → scores more than average; defense < 0 → concedes less)

  Partial pooling:
    Teams with few matches have wider credible intervals.
    [team with most matches] SD = X.XX  vs  [team with fewest] SD = X.XX
```

---

## 5. Constraints and Non-Goals

**Constraints:**
- cl-acorn public API only (`ad:`, `dist:`, `infer:`, `diag:`)
- No external dependencies
- Runtime < 3 minutes on a laptop (reduce chains/samples if needed)
- Same 4-step agent narrative structure as example 12
- Single file: `examples/13-football-hierarchy/main.lisp`

**Non-goals:**
- Posterior predictive simulation (out of scope for this example)
- Time-varying team strengths
- Neutral-ground matches
- Tournament-type effects
- LOO as primary selection criterion (WAIC is sufficient; LOO printed as secondary)

---

## 6. Testing

Manual verification (no automated test file needed for examples):

1. Load `main.lisp` from REPL: `(load "examples/13-football-hierarchy/main.lisp")`
2. Check data load: non-zero match count, all team indices in [0, 14]
3. Check log-posterior finite at initial params: `(funcall lp initial-params)` → finite float
4. Check gradient non-nil: `(ad:gradient lp initial-params)` → list of 35 floats
5. Run with 1 chain × 100 samples as smoke test before full run
6. Verify R-hat < 1.1 for all parameters after full run
7. Verify WAIC values are finite and negative

---

## 7. Key Technical Risks

| Risk | Mitigation |
|------|-----------|
| NUTS slow on 35-dim space | Use 2 chains × 500 samples; step-size auto-adapted |
| R-hat > 1.1 for z_i | Increase n-warmup; check soft constraint sigma |
| Divergences in funnel | Non-centered parameterization already mitigates this |
| `subseq` on AD param list | Verify `subseq` works on lists (not vectors) in CL — it does |
| `nth` with large index on long list | Acceptable for N=15; O(N) but N is tiny |
| Negative p_loo (known issue in diag) | Document as known; doesn't affect model ranking |
