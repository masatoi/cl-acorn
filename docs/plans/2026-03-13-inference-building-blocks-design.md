# 推論アルゴリズムの部品化 — 設計文書

## ゴール

コンセプト文書セクション4「推論アルゴリズムの部品化」を実装する。確率分布、オプティマイザ、HMCサンプラーを「ビルディングブロック」として提供し、AIエージェントがタスクに応じて柔軟にモデリングと推論を組み立てられるようにする。

## スコープ

| コンポーネント | 内容 |
|---------------|------|
| 確率分布 | Normal, Gamma, Beta, Uniform, Bernoulli, Poisson（6種） |
| オプティマイザ | SGD, Adam（2種） |
| 推論 | HMC（ハミルトニアンモンテカルロ） |
| VI（変分推論） | 今回のスコープ外（将来追加可能） |

## パッケージ構成

3つの新しいパッケージを追加。全て `cl-acorn.ad` に依存し、内部で `ad:` 算術を使用する。

| パッケージ | ニックネーム | 役割 |
|-----------|-------------|------|
| `cl-acorn.distributions` | `dist` | 確率分布の log-pdf と sample |
| `cl-acorn.optimizers` | `opt` | 勾配ベース最適化 |
| `cl-acorn.inference` | `infer` | MCMC推論 |

ASDF上は既存の `cl-acorn` システムに新しいモジュールとして追加する。

```
src/
  distributions/
    package.lisp
    util.lisp         — log-gamma等の補助関数
    normal.lisp
    gamma.lisp
    beta.lisp
    uniform.lisp
    bernoulli.lisp
    poisson.lisp
  optimizers/
    package.lisp
    sgd.lisp
    adam.lisp
  inference/
    package.lisp
    hmc.lisp
```

## 確率分布 (`cl-acorn.distributions`)

### API設計

関数ベースAPI。各分布は2つの関数を提供：

- `*-log-pdf`: 対数確率密度（AD対応 — パラメータに dual/tape-node を受け取れる）
- `*-sample`: 乱数生成（通常の double-float を返す）

### 分布一覧

| 分布 | log-pdf 関数 | sample 関数 | パラメータ |
|------|-------------|-------------|-----------|
| Normal | `(dist:normal-log-pdf x :mu mu :sigma sigma)` | `(dist:normal-sample :mu mu :sigma sigma)` | mu: 平均, sigma: 標準偏差 |
| Gamma | `(dist:gamma-log-pdf x :shape k :rate r)` | `(dist:gamma-sample :shape k :rate r)` | shape: 形状, rate: レート |
| Beta | `(dist:beta-log-pdf x :alpha a :beta b)` | `(dist:beta-sample :alpha a :beta b)` | alpha, beta: 形状パラメータ |
| Uniform | `(dist:uniform-log-pdf x :low lo :high hi)` | `(dist:uniform-sample :low lo :high hi)` | low: 下限, high: 上限 |
| Bernoulli | `(dist:bernoulli-log-pdf x :prob p)` | `(dist:bernoulli-sample :prob p)` | prob: 成功確率 |
| Poisson | `(dist:poisson-log-pdf k :rate lam)` | `(dist:poisson-sample :rate lam)` | rate: 平均 |

### 設計原則

1. **`*-log-pdf` は AD 透過的**: 内部で `ad:+`, `ad:*`, `ad:log`, `ad:exp` 等を使用。パラメータが dual なら微分が伝播し、tape-node なら勾配が記録される。
2. **`*-sample` は数値のみ**: 乱数生成は微分不要。CL標準の `random` を使用。
3. **正規化項の扱い**: log-gamma (Stirling近似) は定数項として `cl:` 算術で計算。AD対応不要。
4. **デフォルト値**: 各分布のパラメータにはデフォルト値を設定（例: Normal は mu=0, sigma=1）。

### 補助関数

```lisp
;; src/distributions/util.lisp
(defun log-gammaln (x)
  "Log of the gamma function via Stirling-Lanczos approximation.
   Accepts double-float. Used for normalization constants in gamma/beta distributions.")
```

## オプティマイザ (`cl-acorn.optimizers`)

### API設計

パラメータはリスト `(p1 p2 ... pn)` で表現。関数型スタイル。

### SGD

```lisp
(opt:sgd-step params grads &key (lr 0.01d0))
;; => 更新されたパラメータリスト（新しいリスト）
;;
;; p_i ← p_i - lr * g_i
```

- 副作用なし（新しいリストを返す）
- 既存の例題の手書きSGDを置き換え可能

### Adam

```lisp
;; 状態を初期化
(defvar *state* (opt:make-adam-state n-params))

;; ステップ実行
(opt:adam-step params grads *state*
  &key (lr 0.001d0) (beta1 0.9d0) (beta2 0.999d0) (epsilon 1d-8))
;; => 更新されたパラメータリスト（新しいリスト）
;; *state* は副作用で更新（m, v, step カウンタ）
```

### adam-state 構造体

```lisp
(defstruct adam-state
  (m nil :type list)        ; 1次モーメント推定
  (v nil :type list)        ; 2次モーメント推定
  (step 0 :type fixnum))   ; タイムステップ
```

`make-adam-state` は n-params を受け取り、m と v をゼロで初期化する。

## HMC推論 (`cl-acorn.inference`)

### API

```lisp
(infer:hmc log-pdf-fn initial-params
  &key (n-samples 1000)
       (n-warmup 500)
       (step-size 0.01d0)
       (n-leapfrog 10))
;; => (values samples accept-rate)
;;
;; samples: リストのリスト ((p1 p2 ... pn) ...)  — n-samples 個
;; accept-rate: double-float (0.0 ~ 1.0)
```

### 内部構造

1. **leapfrog 積分器** (`leapfrog-step`):
   - 半ステップ: `p ← p + (ε/2) * ∇log-pdf(q)`
   - 全ステップ: `q ← q + ε * p`
   - 半ステップ: `p ← p + (ε/2) * ∇log-pdf(q)`
   - これを `n-leapfrog` 回繰り返す

2. **勾配計算**: `ad:gradient` を使用。`log-pdf-fn` は1引数（パラメータリスト）を受け取りスカラーを返す関数。

3. **Metropolis-Hastings 判定**:
   - ハミルトニアン: `H(q, p) = -log-pdf(q) + 0.5 * Σ p_i^2`
   - 受容確率: `min(1, exp(H_old - H_new))`

4. **ウォームアップ**: 最初の `n-warmup` 個のサンプルは捨てる（バーンイン）。ステップサイズの自動調整は初回実装では省略（固定ステップサイズ）。

5. **運動量のサンプリング**: 各イテレーションで `N(0,1)` から独立にサンプル。

### 使用例

```lisp
;; データから正規分布のパラメータを推論
(defun model-log-pdf (params)
  "params = (mu, log-sigma)"
  (let ((mu (first params))
        (sigma (ad:exp (second params))))  ; log-sigma → sigma
    (+ (loop for x in *data*
             sum (dist:normal-log-pdf x :mu mu :sigma sigma))
       ;; 事前分布
       (dist:normal-log-pdf mu :mu 0.0d0 :sigma 10.0d0)
       (dist:normal-log-pdf (second params) :mu 0.0d0 :sigma 2.0d0))))

(multiple-value-bind (samples accept-rate)
    (infer:hmc #'model-log-pdf '(0.0d0 0.0d0)
      :n-samples 2000 :n-warmup 500
      :step-size 0.01d0 :n-leapfrog 20)
  (format t "Accept rate: ~,1F%~%" (* 100 accept-rate))
  ;; samples の統計量を計算...
  )
```

## テスト戦略

各パッケージに対応するテストファイルを作成。

| テスト | 内容 |
|-------|------|
| distributions-test | 各分布の log-pdf が既知の値と一致 / AD微分が解析解と一致 / sample が範囲内 |
| optimizers-test | SGD/Adam が既知の二次関数の最小値に収束 |
| hmc-test | 既知の分布（Normal）からのサンプルの平均・分散が理論値と一致 |

## 依存関係

- 外部依存なし（ANSI Common Lisp + cl-acorn.ad のみ）
- サンプリングには CL 標準の `random` を使用
- log-gamma は Lanczos 近似で自前実装
