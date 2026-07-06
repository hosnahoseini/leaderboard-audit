# Clean BT Rank

`clean_bt_rank` is a compact Bradley-Terry / Bradley-Terry-Luce codebase for:

- preparing pairwise preference data,
- fitting a BT model,
- computing several confidence interval backends,
- estimating first-order influence of matches on parameters and objectives,
- turning that influence into action-specific reports for `drop`, `add`, and `flip`,
- running iterative manipulation / robustness procedures,
- and running experiment pipelines on Arena-style data.

## GitHub Pages Website

The static website artifact lives in `docs/index.html`. This repository includes
`.github/workflows/pages.yml`, which publishes `docs/` to GitHub Pages whenever
you push to `master` or `main`.

To publish it on your GitHub account:

```bash
git remote add origin git@github.com:<your-user>/<your-repo>.git
git add README.md .gitignore .github/workflows/pages.yml docs/
git commit -m "Prepare clean GitHub Pages site"
git push -u origin master
```

Then open the repository on GitHub, go to **Settings -> Pages**, and set the
source to **GitHub Actions**.

This README explains the codebase as it is actually implemented: what each file
does, what algorithm is used there, and how the main pieces fit together.

## Repository Structure

Main package files:

- [src/clean_bt_rank/datasets.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/datasets.py)
- [src/clean_bt_rank/bt_model.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/bt_model.py)
- [src/clean_bt_rank/ci.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/ci.py)
- [src/clean_bt_rank/parameter_influence.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/parameter_influence.py)
- [src/clean_bt_rank/objectives.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/objectives.py)
- [src/clean_bt_rank/actions.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/actions.py)
- [src/clean_bt_rank/iterative_actions.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/iterative_actions.py)
- [src/clean_bt_rank/plotting.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/plotting.py)
- [src/clean_bt_rank/reporting.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/reporting.py)

Experiment modules:

- [src/clean_bt_rank/experiments/ci_reduction.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/ci_reduction.py)
- [src/clean_bt_rank/experiments/topk_manipulation.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/topk_manipulation.py)
- [src/clean_bt_rank/experiments/_common.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/_common.py)

Notebooks:

- [notebooks/arena_bt_influence_analysis.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_bt_influence_analysis.ipynb)
- [notebooks/arena_iterative_actions_paper.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_iterative_actions_paper.ipynb)
- [notebooks/arena_ci_reduction.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_ci_reduction.ipynb)
- [notebooks/arena_topk_manipulation.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_topk_manipulation.ipynb)
- [notebooks/arena_experiment_pipelines.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_experiment_pipelines.ipynb) (index → links above)

Tests:

- [tests/test_workflow.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/tests/test_workflow.py)
- [tests/verify_against_baseline.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/tests/verify_against_baseline.py)

## High-Level Design

The codebase is built in layers.

1. `datasets.py`
   Turn raw match data into a standard BT design matrix.
2. `bt_model.py`
   Fit the BT model once and cache the fitted quantities needed later.
3. `ci.py`
   Compute confidence intervals and standard errors from the fitted model.
4. `parameter_influence.py`
   Estimate how the fitted parameter vector changes when data is dropped or added.
5. `objectives.py`
   Define scalar functions of the fitted skill vector.
6. `actions.py`
   Convert parameter influence into objective influence under `drop`, `add`, or `flip`.
7. `iterative_actions.py`
   Repeatedly apply actions until a target condition is met.
8. `src/clean_bt_rank/experiments/*`
   Use the above layers to run higher-level experiments on Arena.

The main design principle is separation of concerns:

- model fitting does not hardcode any objective,
- objectives do not know about Hessians or data actions,
- action logic does not hardcode a specific objective,
- experiments use the framework rather than duplicating it.

## End-to-End Workflow

The typical workflow is:

1. Load a dataframe of pairwise matches.
2. Build a [BattleDataset](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/datasets.py).
3. Fit a [BradleyTerryModel](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/bt_model.py).
4. Compute:
   - ratings and CIs,
   - parameter influence,
   - objective influence,
   - action influence,
   - or iterative interventions.

Conceptually:

```text
raw dataframe
  -> BattleDataset
  -> BradleyTerryModel.fit()
  -> CI / objectives / parameter influence
  -> action reports
  -> iterative procedures / experiments
```

## Data Layer

Implemented in:

- [datasets.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/datasets.py)

### What it does

This file converts human-readable pairwise data into the numeric objects the BT
model needs:

- a list of competitor names,
- integer-encoded player pairs,
- numeric outcomes,
- a design matrix `X`.

### Outcome mapping

The raw dataframe may use labels such as:

- `model_a`
- `model_b`
- `tie`
- `both_bad`

These are mapped to numbers in `[0, 1]` using `_map_outcome(...)`.

The internal semantic meaning is:

- `1.0`: the row’s `model_a` wins,
- `0.0`: the row’s `model_b` wins,
- `0.5`: a tie before tie expansion.

### Tie handling algorithm

This codebase uses one tie-handling rule everywhere: weighted symmetric
expansion.

Suppose an original comparison is

$$
z = (a, b, y), \qquad y \in \{0, 0.5, 1\}.
$$

Then `_build_weighted_symmetric_frame(...)` creates two rows:

1. forward row `(a, b)`
2. reverse row `(b, a)`

with outcomes:

$$
y_{\text{forward}} = y, \qquad y_{\text{reverse}} = 1-y
$$

for non-ties, and

$$
y_{\text{forward}} = y_{\text{reverse}} = 1
$$

for ties.

So the model never sees a `0.5` outcome after dataset construction. This is why
the BT fitting code can safely assume binary outcomes.

### Competitor indexing algorithm

`BattleDataset.from_dataframe(...)` builds the competitor list by collecting all
unique names appearing in `model_a` or `model_b`, unless a competitor list is
passed explicitly.

Then every row is encoded as:

$$
(\text{player\_a\_index}, \text{player\_b\_index}).
$$

This yields:

- `competitors`: ordered list of names,
- `pairs`: integer pair array,
- `outcomes`: binary outcome array,
- `frame`: cleaned expanded dataframe.

### Design matrix algorithm

`BattleDataset.design_matrix()` builds a reference-player design matrix.

If there are `n` competitors, the BT model uses `p = n - 1` free parameters.
Player `0` is the reference by default.

For each expanded row comparing `a` and `b`:

- put `+1` in the column for `a` if `a` is not the reference,
- put `-1` in the column for `b` if `b` is not the reference,
- leave the reference player implicit.

This ensures:

$$
x_i^\top \beta = \theta_{a_i} - \theta_{b_i}.
$$

### Synthetic data helpers

`generate_synthetic_dataframe(...)` and `generate_synthetic_dataset(...)` create
simulated pairwise data by:

1. sampling true skills,
2. sampling matchups,
3. sampling winners from the BT probability,
4. optionally inserting ties before weighted symmetric expansion.

These are mainly used for testing and sanity checks.

## Bradley-Terry Model Layer

Implemented in:

- [bt_model.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/bt_model.py)

### What it does

This class fits the BT model and stores all fitted quantities needed by the rest
of the framework.

### Statistical model

For players `a` and `b`:

$$
\Pr(a \text{ beats } b) = \sigma(\theta_a - \theta_b),
$$

where

$$
\sigma(t) = \frac{1}{1 + e^{-t}}.
$$

Using the reference-player parameterization:

$$
\beta \in \mathbb{R}^{n-1},
$$

and the reference player has skill `0`.

### Fitting algorithm

The code fits the binary BT likelihood using:

- `sklearn.LogisticRegression(fit_intercept=False, penalty=None)`

with the BT design matrix `X` and binary outcomes `y`.

This is equivalent to fitting the logistic negative log-likelihood

$$
\ell(\beta)
=
\sum_{i=1}^m \Bigl[\log(1 + e^{x_i^\top \beta}) - y_i x_i^\top \beta\Bigr].
$$

### Quantities cached after fit

After fitting, the model stores:

- `beta_hat_`
  free parameter vector
- `full_beta_hat_`
  full skill vector with the reference entry inserted
- `reported_skills_`
  current full skill vector used by objectives and CIs
- `probabilities_`
  fitted row probabilities
- `residuals_ = y - p`
- `hessian_reg_`
  Hessian plus optional ridge
- `solve_h_xt_`
  the matrix `H^{-1} X^T`, transposed row-wise for reuse
- `leverage_`
  row leverage values for 1sN

### Hessian algorithm

At the fitted point:

$$
p_i = \sigma(x_i^\top \hat\beta), \qquad v_i = p_i (1 - p_i).
$$

The Hessian is:

$$
H = X^\top \operatorname{diag}(v) X.
$$

The code stores:

$$
H_{\text{reg}} = H + \lambda I,
$$

where `lambda = hessian_ridge`.

Then it computes:

$$
H_{\text{reg}}^{-1} X^\top
$$

once and reuses it in influence code.

### Reference player and scaling

The model is fit in natural BT coordinates, but displayed scores can be shifted
to match Arena-style reporting:

$$
\text{score}_i = \text{INIT\_RATING} + \alpha \theta_i + \text{ANCHOR\_SHIFT},
$$

where

$$
\alpha = \frac{\text{scale}}{\log(\text{base})}.
$$

If `anchor_player` and `anchor_rating` are supplied, the code computes a single
constant shift so that the chosen player is displayed at the chosen score.

This transformation does not change:

- rankings,
- pairwise skill gaps,
- probabilities,
- influence calculations in natural BT space.

### Pair count algorithm

`observed_pair_counts()` reconstructs a symmetric count matrix `A` from the
fitted design rows. This is used by the Gao/local CI backend and by uncertainty
objectives.

## Confidence Intervals

Implemented in:

- [ci.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/ci.py)

### What it does

This file provides a unified CI interface and multiple CI backends. Every
backend returns a `CIResult` with the same fields:

- method name,
- alpha,
- point estimate,
- standard error,
- lower bound,
- upper bound.

### Shared API

Main entry point:

```python
compute_confidence_intervals(model, method="sandwich", alpha=0.05, **kwargs)
```

There is also a lower-level SE helper:

```python
compute_standard_errors(model, method="gao_local")
```

which returns natural-scale SEs and, when implemented, SE gradients.

### Sandwich CI

Algorithm:

1. Compute row score vectors:

$$
u_i = x_i (y_i - p_i).
$$

2. Build the empirical meat:

$$
M = \sum_i u_i u_i^\top = U^\top U.
$$

3. Compute the free-parameter sandwich covariance:

$$
\Sigma_{\text{free}} = H^{-1} M H^{-1}.
$$

4. Expand it to the full skill vector by inserting the reference coordinate.
5. Use the diagonal to form Wald intervals.

The interval is

$$
\hat\theta_i \pm z_{1-\alpha/2} \cdot \operatorname{se}_i.
$$

In code this is done by `_compute_sandwich_covariance(...)`.

### Bootstrap CI

Algorithm:

1. Sample rows with replacement from the fitted dataset.
2. Refit the BT model on each bootstrap sample.
3. Collect the resulting skill or rating samples.
4. Use:
   - empirical standard deviation for `standard_error`,
   - empirical quantiles for `lower` and `upper`.

This is implemented by `_bootstrap_skill_samples(...)` and
`_bootstrap_rating_samples(...)`.

Important note:

- the bootstrap backend is used as a reporting CI backend,
- but it does not currently provide a differentiable SE gradient.

### Gao / local asymptotic CI

This backend uses the sample analogue of the coordinate-wise information scale.

For player `i`:

$$
\rho_i(\theta)^2 = \sum_{j \ne i} A_{ij} \, \sigma(\theta_i - \theta_j)\bigl(1-\sigma(\theta_i - \theta_j)\bigr).
$$

Then

$$
\operatorname{se}_i = \frac{1}{\rho_i(\hat\theta)}.
$$

The CI is

$$
\hat\theta_i \pm z_{1-\alpha/2} \operatorname{se}_i.
$$

This backend is especially important in this codebase because it also provides
the gradient of `se_i` with respect to the full BT skill vector. That gradient
is used by the CI-aware boundary objective.

### CI backend support for gradients

Currently:

- `gao_local`: SE values and SE gradients available,
- `sandwich`: SE values available, no SE gradient,
- `bootstrap`: SE values available, no SE gradient.

That behavior is used explicitly in `CIBoundaryObjective`.

## Parameter Influence

Implemented in:

- [parameter_influence.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/parameter_influence.py)

### What it does

This class estimates how the fitted parameter vector changes when a row is:

- deleted,
- or hypothetically added.

It operates entirely at the parameter level. It does not know anything about a
specific downstream objective.

### Row score contribution

For row `i`:

$$
s_i(\hat\beta) = x_i (y_i - p_i).
$$

This is returned by `score_contributions()`.

### Deletion IF algorithm

The first-order deletion approximation is:

$$
\Delta \beta_i^{\mathrm{IF}} \approx - H^{-1} s_i(\hat\beta).
$$

In code:

- `parameter_change_if()`

returns the estimated parameter change **after deleting row `i`**.

Because `solve_h_xt_` already stores `H^{-1} X^T`, the implementation becomes:

$$
\Delta \beta_i^{\mathrm{IF}}
=
-(y_i - p_i)\,(H^{-1}x_i).
$$

### Deletion 1sN algorithm

The one-step Newton correction is:

$$
\Delta \beta_i^{\mathrm{1sN}}
\approx
\frac{\Delta \beta_i^{\mathrm{IF}}}{1 - h_i},
$$

where

$$
h_i = v_i x_i^\top H^{-1} x_i.
$$

In code:

- `parameter_change_1sn()`

### Addition IF algorithm

For a hypothetical new row `(x_{\text{new}}, y_{\text{new}})`, the first-order
change is:

$$
\Delta \beta_{\text{add}}^{\mathrm{IF}}
\approx
H^{-1} x_{\text{new}} (y_{\text{new}} - p_{\text{new}}),
$$

with

$$
p_{\text{new}} = \sigma(x_{\text{new}}^\top \hat\beta).
$$

In code:

- `candidate_parameter_change_if(...)`

### Addition 1sN algorithm

The additive one-step Newton correction is:

$$
\Delta \beta_{\text{add}}^{\mathrm{1sN}}
\approx
\frac{\Delta \beta_{\text{add}}^{\mathrm{IF}}}{1 + h_{\text{new}}}.
$$

In code:

- `candidate_parameter_change_1sn(...)`

### Return shapes

The parameter influence API returns either:

- `(n_rows, p)` when `dim=None`,
- or `(n_rows,)` for one selected coordinate.

## Objective Layer

Implemented in:

- [objectives.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/objectives.py)

### What it does

This file defines scalar functions of the fitted skill vector. Each objective
implements:

- `value(bt_model)`
- `gradient_free(bt_model)`
- `name`

So every objective can plug into the same influence machinery.

### Shared objective-influence decomposition

For any scalar objective `f`, the code computes rowwise action influence as

$$
\Delta f_r
\approx
\nabla_\beta f(\hat\beta)^\top \Delta \beta_r
+
\frac{\partial f(\hat\beta, w)}{\partial w_r}.
$$

The two pieces are:

1. parameter-mediated term

$$
\Delta f_r^{(\theta)} = \nabla_\beta f(\hat\beta)^\top \Delta \beta_r,
$$

2. explicit data-weight term

$$
\Delta f_r^{(\mathrm{explicit})}
=
\frac{\partial f(\hat\beta, w)}{\partial w_r}.
$$

So

$$
\Delta f_r \approx \Delta f_r^{(\theta)} + \Delta f_r^{(\mathrm{explicit})}.
$$

For the BT parameter change, the implementation uses:

$$
\Delta \beta_r^{\mathrm{drop,IF}} \approx -H^{-1} s_r,
\qquad
\Delta \beta_r^{\mathrm{drop,1sN}} \approx \frac{-H^{-1} s_r}{1-h_r},
$$

$$
\Delta \beta_r^{\mathrm{add,IF}} \approx +H^{-1} s_r,
\qquad
\Delta \beta_r^{\mathrm{add,1sN}} \approx \frac{+H^{-1} s_r}{1+h_r},
$$

where `s_r = x_r (y_r - p_r)` and `h_r = v_r x_r^\top H^{-1} x_r` with
`v_r = p_r(1-p_r)`.

### Skill gap objective

For players `a` and `b`:

$$
f(\theta) = \theta_a - \theta_b.
$$

The full gradient is:

$$
\nabla f = e_a - e_b.
$$

Its influence decomposition is therefore

$$
\Delta f_r^{(\theta)}
=
(e_a-e_b)^\top \Delta\beta_r,
$$

and the explicit term is identically zero:

$$
\Delta f_r^{(\mathrm{explicit})} = 0
\qquad
\text{for drop, add, and flip.}
$$

### Player uncertainty objective

This is an information-based variance proxy:

$$
f_i(\theta) = \frac{1}{\rho_i(\theta)^2},
$$

where

$$
\rho_i(\theta)^2 = \sum_{j \ne i} A_{ij} \, \sigma(\theta_i-\theta_j)\bigl(1-\sigma(\theta_i-\theta_j)\bigr).
$$

Let

$$
v_{ij} = \sigma(\theta_i-\theta_j)\bigl(1-\sigma(\theta_i-\theta_j)\bigr),
\qquad
v'_{ij} = v_{ij}\bigl(1-2\sigma(\theta_i-\theta_j)\bigr).
$$

Then

$$
\frac{\partial f_i}{\partial \theta_k}
=
-\frac{1}{(\rho_i^2)^2}
\frac{\partial \rho_i^2}{\partial \theta_k},
$$

with

$$
\frac{\partial \rho_i^2}{\partial \theta_k}
=
\sum_{j \ne i} A_{ij} v'_{ij}\,(\mathbf{1}\{k=i\}-\mathbf{1}\{k=j\}).
$$

So the gradient-mediated influence is

$$
\Delta f_r^{(\theta)}
=
\nabla_\beta f_i(\hat\beta)^\top \Delta\beta_r.
$$

For a row `r` comparing players `u_r` and `v_r`, the explicit term used by the
code is:

$$
\Delta f_r^{(\mathrm{explicit})}
=
\operatorname{sign}(\text{action})
\left(
-\frac{v_r \,\mathbf{1}\{i \in \{u_r,v_r\}\}}{(\rho_i^2)^2}
\right),
$$

where `sign(drop) = -1`, `sign(add) = +1`, and for `flip` the explicit term is
set to zero:

$$
\Delta f_r^{(\mathrm{explicit})} = 0
\qquad
\text{for flip.}
$$

### Trace uncertainty objective

This is the global proxy:

$$
f(\theta) = \sum_i \frac{1}{\rho_i(\theta)^2}.
$$

Its gradient is the sum of the playerwise gradients:

$$
\nabla f(\theta) = \sum_i \nabla \left(\frac{1}{\rho_i(\theta)^2}\right).
$$

Hence

$$
\Delta f_r^{(\theta)}
=
\nabla_\beta f(\hat\beta)^\top \Delta\beta_r.
$$

For a row `r` comparing players `u_r` and `v_r`, the explicit term is

$$
\Delta f_r^{(\mathrm{explicit})}
=
\operatorname{sign}(\text{action})
\left(
-v_r
\left[
\frac{1}{(\rho_{u_r}^2)^2}
+
\frac{1}{(\rho_{v_r}^2)^2}
\right]
\right),
$$

again with `sign(drop) = -1`, `sign(add) = +1`, and

$$
\Delta f_r^{(\mathrm{explicit})} = 0
\qquad
\text{for flip.}
$$

### Kendall tau surrogate objective

This defines a smooth ranking-consistency objective relative to a fixed
reference ranking:

$$
f_{\tau,T}(\theta; \pi)
=
\frac{2}{P(P-1)}
\sum_{a<b}
s_{ab}\,
\tanh\!\left(\frac{\theta_a - \theta_b}{T}\right),
$$

where `s_ab` encodes the reference ordering.

Algorithm:

1. build or read the sign matrix,
2. evaluate pairwise `tanh` terms,
3. sum over unordered pairs,
4. differentiate using

$$
\frac{d}{dx}\tanh(x/T) = \frac{1}{T}\bigl(1-\tanh^2(x/T)\bigr).
$$

The full gradient is

$$
\frac{\partial f_{\tau,T}}{\partial \theta_k}
=
\frac{2}{P(P-1)}
\sum_{a<b}
s_{ab}
\frac{1-\tanh^2((\theta_a-\theta_b)/T)}{T}
(\mathbf{1}\{k=a\}-\mathbf{1}\{k=b\}).
$$

So

$$
\Delta f_r^{(\theta)}
=
\nabla_\beta f_{\tau,T}(\hat\beta)^\top \Delta\beta_r.
$$

Its explicit term is identically zero:

$$
\Delta f_r^{(\mathrm{explicit})} = 0
\qquad
\text{for drop, add, and flip.}
$$

### CI-aware top-k boundary objective

For the frozen rank-`k` and rank-`k+1` players, `a` and `b`, define

$$
f_{\mathrm{CI}}(\theta)
=
\bigl(\theta_a - z s_a(\theta)\bigr)
-
\bigl(\theta_b + z s_b(\theta)\bigr).
$$

Interpretation:

- if this is positive, the top-k boundary is CI-separated,
- if it is nonpositive, uncertainty closes or reverses that margin.

Algorithm:

1. when the objective is constructed, freeze the current rank-`k` and
   rank-`k+1` players,
2. choose a CI backend,
3. compute the value using the current skills and SEs,
4. compute the gradient

$$
\nabla f_{\mathrm{CI}}
=
e_a - e_b - z \nabla s_a(\theta) - z \nabla s_b(\theta).
$$

In code, this objective requires a CI backend that exposes SE gradients. If the
backend does not provide them, it raises an error instead of silently falling
back.

For `gao_local` and `local_asymptotic`, the code also includes an explicit
weight term. Writing `s_i(\theta) = 1/\sqrt{\rho_i^2}`, we have

$$
\frac{\partial s_i}{\partial w_r}
=
-\frac{1}{2}
\frac{v_r \,\mathbf{1}\{i \in \{u_r,v_r\}\}}{(\rho_i^2)^{3/2}}.
$$

Therefore

$$
\Delta f_r^{(\mathrm{explicit})}
=
\operatorname{sign}(\text{action})
\left(
-z \frac{\partial s_a}{\partial w_r}
-z \frac{\partial s_b}{\partial w_r}
\right),
$$

with `sign(drop) = -1`, `sign(add) = +1`, and

$$
\Delta f_r^{(\mathrm{explicit})} = 0
\qquad
\text{for flip.}
$$

For other CI backends, the explicit term is zero because the base
implementation is used.

### Objective influence

`ObjectiveInfluence` combines a parameter influence object with an objective.

For each row:

$$
\Delta f_r
\approx
\nabla_\beta f(\hat\beta)^\top \Delta \beta_r
+
\frac{\partial f(\hat\beta, w)}{\partial w_r}.
$$

So objectives with no explicit term use only the gradient term, and objectives
with a nonzero explicit term use both pieces.

## Action Layer

Implemented in:

- [actions.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/actions.py)

### What it does

This file converts objective influence into action-specific reports for:

- `drop`
- `add`
- `flip`

The public API is:

```python
compute_influence(bt_model, objective, action="drop", method="if")
```

### Drop algorithm

1. compute parameter deletion influence,
2. project through the objective gradient,
3. attach the original row metadata.

The output contains one row per existing fitted row.

### Add algorithm

1. generate candidate hypothetical rows,
2. compute parameter addition influence for each candidate,
3. project through the objective gradient,
4. optionally weight by candidate BT probability.

Candidate modes:

- `all_outcomes`
  every ordered pair
- `all_pairs`
  one ordered candidate per unordered pair, using the expected winner
- `all_outcomes_weighted`
  same as `all_outcomes`, but final influence is multiplied by fitted outcome probability

### Flip algorithm

Flip is implemented as:

$$
\Delta f_{\text{flip}} \approx \Delta f_{\text{add flipped}} - \Delta f_{\text{drop}}.
$$

Algorithm:

1. compute the drop report on the existing row,
2. build the same row with outcome `1-y`,
3. compute the add report for that flipped row,
4. subtract.

This keeps `flip` symmetric with the other two actions and avoids duplicate
mathematics.

## Iterative Intervention Layer

Implemented in:

- [iterative_actions.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/iterative_actions.py)
- [iterative_dropping.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/iterative_dropping.py)

### What it does

This layer repeatedly applies the most promising action until a condition is met.

Main API:

```python
run_iterative_action_until_target(...)
run_iterative_actions_for_objectives(...)
```

### Iterative algorithm

For a chosen objective, action, and stopping condition:

1. evaluate current objective value,
2. compute current action-level influence report,
3. filter to available candidates,
4. score each candidate by reduction in condition distance,
5. apply the best candidate,
6. either:
   - refit and reevaluate exactly, or
   - update approximately using predicted influence,
7. repeat until success or budget exhaustion.

### Check modes

`refit`

- physically update the dataset,
- refit the BT model,
- recompute the objective exactly.

`approximate`

- do not refit after each step,
- accumulate the predicted objective change directly.

This makes the framework useful both for exact but slower intervention studies
and for cheap first-order approximations.

### Built-in utilities

`boundary_gap_objective(...)`

- builds a skill-gap objective at the top-k boundary.

`top_k_swap_objectives(...)`

- builds several inside/outside top-k gap objectives near the boundary.

`make_nonpositive_condition()`

- common stopping rule for “drive the objective to zero or below”.

## Reporting and Plotting

Implemented in:

- [reporting.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/reporting.py)
- [plotting.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/plotting.py)

### Reporting

The reporting utilities mostly:

- convert influence arrays into dataframes,
- sort by positive / negative / absolute influence,
- preserve match metadata for inspection.

### Plotting

The plotting file provides reusable visualization helpers for:

- rating plots with CIs,
- CI method comparison,
- objective trajectories,
- iterative action trajectories,
- predicted vs actual curves,
- experiment curves,
- ranking before/after plots.

These are used by the Arena notebooks and experiment notebooks to keep the
analysis code cleaner.

## Experiment Pipelines

Implemented in:

- [experiments/ci_reduction.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/ci_reduction.py)
- [experiments/topk_manipulation.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/topk_manipulation.py)
- [experiments/_common.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/_common.py)

### Shared experiment helper

`_common.py` contains the state-management logic used by experiments:

- build a mutable experiment state from a fitted model,
- apply add/drop/flip edits to that state,
- refit a fresh BT model from the edited state,
- extract ranking dataframes.

This prevents the experiment modules from duplicating state-update code.

### CI reduction experiment

Goal:

- reduce the uncertainty or CI width of one chosen player.

Important design choice:

- this experiment uses **add** actions, not drop, because dropping data would
  usually increase uncertainty.

Objective used:

- `PlayerUncertaintyObjective(target_player)`

Policies implemented:

1. `influence`
   compute one initial add-action influence ranking on the original model and
   then greedily walk that presorted list.
2. `random`
   randomly choose among all available candidate additions.
3. `active_ranking`
   choose the candidate with largest fitted logistic variance
   `p(1-p)`.
4. `arena_active`
   use the Arena-style pair-variance / pair-count baseline implemented in
   [arena_active_sampling.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/src/clean_bt_rank/experiments/arena_active_sampling.py).

The `active_ranking` policy is a practical uncertainty-sampling baseline
inspired by active ranking / active sampling ideas; it is not a claim of exact
paper reproduction.

Candidate modes implemented for add actions:

- `all_pairs`
  one expected-winner row per unordered pair
- `all_outcomes`
  both ordered outcomes for every pair
- `all_outcomes_weighted`
  same candidate set as `all_outcomes`, but the final influence score is
  multiplied by the fitted win probability

The Arena baseline always uses its own `all_pairs`-style candidate generation
internally.

How to compute the compact paper table comparing action spaces:

- run the CI-reduction benchmark separately for each candidate mode
  (`all_outcomes`, `all_outcomes_weighted`, `all_pairs`)
- for each dataset, evaluate three target players at the 10th, 50th, and 90th
  rank quantiles
- read per-target nAUC values from
  `notebooks/artifacts/.../task_metrics.csv`
- for `influence` and `arena_active`, each target contributes one nAUC value
- for `random`, first average nAUC across its 30 trials for each target, then
  use that target-level mean
- compute one dataset-level nAUC per policy by averaging over the three target
  quantiles
- convert those dataset-level nAUC values into relative improvement percentages
  using

  `100 * (baseline_nauc - influence_nauc) / baseline_nauc`

- report each table cell as

  `(+ improvement vs arena_active) / (+ improvement vs random)`

Example: if a dataset has

- `Influence nAUC = 0.9227`
- `Arena Active nAUC = 0.9657`
- `Random nAUC = 0.9802`

then the compact table entry is

- vs Arena Active:
  `100 * (0.9657 - 0.9227) / 0.9657 = 4.44%`
- vs Random:
  `100 * (0.9802 - 0.9227) / 0.9802 = 5.87%`

The overall `All datasets` row is not the mean of the displayed dataset rows.
It is computed by averaging target-level improvements over all valid targets in
the run. In the current CI-reduction artifacts this total is 19 targets, as
recorded by the `all` row in
`notebooks/artifacts/.../dataset_improvement_table.csv`.

Algorithm:

1. build the initial BT model,
2. build the add candidate pool,
3. score candidates by policy,
4. add one candidate row,
5. refit the BT model,
6. compute the target player’s CI width and uncertainty proxy,
7. repeat for the budget.

Outputs:

- per-step history,
- selected action rows,
- summary tables,
- policy-level benchmark histories.

### Top-k manipulation experiment

Goal:

- promote a non-top-k player into top-k,
- or demote a top-k player outside top-k.

Actions supported:

- `add`
- `drop`
- `flip`

Policies:

1. `influence`
2. `random`

Objective used:

- a target-vs-boundary skill gap, constructed dynamically from the current
  ranking.

For example, in promotion mode the objective is a gap between the target player
and the current top-k boundary player.

Add candidate modes:

- `all_outcomes`
- `all_outcomes_weighted`
- `all_pairs`

Algorithm:

1. fit the BT model,
2. build the current target-vs-boundary gap objective,
3. compute action influence for the chosen action type,
4. score all available candidates using the current boundary objective,
5. choose by influence or random policy,
6. apply the action,
7. refit,
8. check whether the target player crossed the top-k boundary.

The random baseline samples from the same full candidate space as the
influence-based method, but without using influence information.

Outputs:

- per-step histories,
- selected action sequences,
- before/after rankings,
- summary tables,
- random-trial benchmark traces for each action.

## Arena Notebooks

### Arena BT influence analysis

- [arena_bt_influence_analysis.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_bt_influence_analysis.ipynb)

This notebook:

- loads a named Arena-style dataset,
- runs a compact EDA pass before fitting BT,
- fits the BT model and saves a leaderboard figure,
- builds the default objective set:
  - `skill_gap`
  - `ci_boundary`
  - `player_uncertainty`
  - `trace_uncertainty`
  - `kendall_tau`
- computes influence reports for:
  - `drop`
  - `flip`
  - `add_all_outcomes`
  - `add_all_pairs`
  - `add_weighted`
- saves one CSV per dataset / action / objective / method combination,
- plots top influential items by action,
- augments reports with match-level covariates such as:
  - match count
  - bridge variance
  - closeness log gap
  - surprise
- summarizes how `|influence|` correlates with those covariates,
- measures whether top influential matches contain the objective’s focus
  player(s),
- saves:
  - overall correlation heatmaps,
  - focus / Hit@K plots,
  - per-objective Spearman heatmaps,
  - per-dataset correlation summaries across every registered dataset.

Typical artifact names are in
[notebooks/artifacts](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/artifacts),
for example:

- `arena55k__drop__skill_gap__...__1sn.csv`
- `arena55k__influence_covariate_correlation__1sn.png`
- `arena55k__influence_focus_hit_at_20__1sn.png`
- `all_datasets__influence_covariate_correlation__1sn.csv`

### Arena CI reduction experiment

- [arena_ci_reduction.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_ci_reduction.ipynb)

This notebook:

- loads Chatbot Arena 55k,
- selects a mid-table target player with meaningful uncertainty,
- runs the CI reduction benchmark using:
  - `influence`
  - `arena_active`
  - multiple influence candidate modes (`all_pairs`, `all_outcomes`,
    `all_outcomes_weighted`)
- tracks CI width, normalized CI width, standard error, variance proxy, rating,
  and rank over the intervention budget,
- compares CI-width reduction curves,
- checks how often selected matches directly involve the target player,
- saves summary and history CSVs plus paper-style figures.

Saved artifacts include:

- `arena_experiments/ci_reduction_summary.csv`
- `arena_experiments/ci_reduction_history.csv`
- `arena_experiments/ci_width_vs_matches.png`
- `arena_experiments/ci_target_involvement.png`

### Arena top-k manipulation experiment

- [arena_topk_manipulation.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_topk_manipulation.ipynb)

This notebook:

- loads Chatbot Arena 55k,
- identifies near-boundary promotion and demotion targets from the fitted
  ranking,
- runs influence-guided and random top-k manipulation experiments,
- focuses on add-action interventions in the notebook figures,
- records whether promotion / demotion succeeds within budget,
- plots ranking before vs after for the influence-selected intervention,
- compares matches needed under influence vs random,
- computes a rigging-exposure comparison,
- plots skill-gap trajectories during promotion / demotion,
- saves CSV summaries and figure outputs.

Saved artifacts include:

- `arena_experiments/topk_summary.csv`
- `arena_experiments/ranking_before_after_promote.png`
- `arena_experiments/ranking_before_after_demote.png`
- `arena_experiments/topk_matches_needed.png`
- `arena_experiments/influence_vs_rigging.png`
- `arena_experiments/skill_gap_promote.png`
- `arena_experiments/skill_gap_demote.png`

### Arena robustness / actions-needed notebook

- [arena_robustness_actions_needed.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/arena_robustness_actions_needed.ipynb)

This notebook is the broadest iterative-action analysis. It:

- studies how many actions are needed to change top-1 or top-k boundary status,
- compares `drop`, `flip`, `add_pairs`, `add_outcomes`, and `add_weighted`,
- evaluates both point-estimate boundary objectives and CI-aware boundary
  objectives,
- builds greedy influence orderings and random orderings,
- plots Kendall tau degradation curves,
- plots CI-boundary-gap degradation curves,
- exports ranking before/after figures for the strongest interventions,
- can batch-export action-needed summaries across all registered datasets.

It writes outputs under
[notebooks/artifacts/robustness](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/artifacts/robustness),
including per-dataset CSV summaries and PDF/PNG plot pairs.

#### How `k` is chosen in the CI-aware vs non-CI comparison

In the `ci_vs_nonci_actions_needed` pipeline, `k` is chosen once from the CI
ranking and then reused for both methods. The point-estimate (`nonci`) run does
not choose a separate `k`.

The purpose is to compare two success criteria at the same top-`k` boundary:

- point-estimate success: the target pair reverses by the fitted rating gap,
- CI-aware success: the target pair reverses and becomes CI-separated in the
  new order.

The implemented `k`-selection rule is:

1. compute the ranked CI table for all players,
2. for every possible boundary `k`, look at the adjacent pair consisting of the
   rank-`k` player and the rank-`k+1` player,
3. measure the amount of CI overlap at that adjacent boundary,
4. count how many cross-boundary top-`k` vs outside-`k` pairs have overlapping
   CIs,
5. choose `k` by the following priority:
   - maximize adjacent boundary overlap,
   - then maximize the number of eligible overlapping cross-boundary pairs,
   - then minimize the point-estimate boundary gap,
   - then prefer smaller `k`.

If there is no adjacent boundary overlap, the code falls back to the boundary
with the most overlapping cross-boundary pairs. If no overlap exists at all, it
falls back to the boundary with the smallest adjacent point-estimate gap.

After this CI-based boundary is chosen, the non-CI run reuses:

- the same `k`,
- the same boundary top player,
- the same boundary outside player.

So the CI-aware vs non-CI comparison is not comparing two separately optimized
top-`k` choices. It is comparing two attack objectives on the same ambiguous
boundary.

### Player-level Kendall tau influence notebook

- [player_kendall_tau_influence_analysis.ipynb](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/notebooks/player_kendall_tau_influence_analysis.ipynb)

This notebook adds a player-level view on top of the match-level Kendall tau
influence pipeline. It:

- aggregates match influence into player influence statistics,
- computes signed, absolute, and joint-Newton player influence summaries,
- correlates those player influence scores with player-level metrics,
- compares approximate player influence against exact refit-based ablations for
  extreme-skill players,
- analyzes ranking shift after dropping the most influential player,
- saves scatter plots, top-player bar charts, ablation comparison plots, and
  rank-shift figures.

Example artifacts:

- `arena55k_player_kendall_influence_scatter_grid.png`
- `arena55k_player_kendall_influence_top_players.png`
- `arena55k_player_kendall_ablation_comparison.png`
- `arena55k_player_kendall_most_influential_rank_shift.png`
- `arena55k__player_kendall_joint_newton_influence_metric_correlations.png`

## Validation

Main validation files:

- [tests/test_workflow.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/tests/test_workflow.py)
- [tests/verify_against_baseline.py](/Users/hoyarhos/Desktop/IF_framework/clean_bt_rank/tests/verify_against_baseline.py)

What is checked:

- dataset construction,
- BT fitting,
- CI output shapes,
- objective gradients against finite differences,
- action symmetry,
- iterative action sanity,
- experiment pipeline sanity,
- exact Arena agreement with the trusted baseline script for IF and 1sN.

The baseline verification is especially important: it confirms that the modular
implementation reproduces the trusted Arena influence code on real Arena data.

## Practical Notes

### What is exact vs approximate

Exact:

- BT fitting,
- CI reporting,
- iterative procedures in `refit` mode.

Approximate / first-order:

- IF and 1sN parameter influence,
- action influence,
- iterative procedures in `approximate` mode.

### What is intentionally frozen

Several objectives intentionally freeze some reference structure:

- `KendallTauObjective` can freeze a ranking,
- `CIBoundaryObjective` freezes the rank-`k` / rank-`k+1` pair,
- experiment targets are often chosen from the initial fit.

This makes the objectives stable enough for first-order influence analysis.

### Main assumptions

- tie handling is always weighted symmetric,
- BT outcomes are binary after expansion,
- the reference-player parameterization is used throughout,
- `CIBoundaryObjective` requires a CI backend that exposes SE gradients,
- notebook analyses reuse cached artifact CSVs when available in several
  places, especially the influence-analysis passes across datasets.

## Minimal Usage Examples

### Fit BT and compute CIs

```python
from clean_bt_rank import BattleDataset, BradleyTerryModel

dataset = BattleDataset.from_dataframe(df)
model = BradleyTerryModel.from_dataset(dataset).fit()
summary = model.summary(ci_method="gao_local")
```

### Influence on a skill gap

```python
from clean_bt_rank import BTParameterInfluence, ObjectiveInfluence, SkillGapObjective

param_infl = BTParameterInfluence(model)
gap_obj = SkillGapObjective("gpt-4-0613", "gpt-4-0314")
obj_infl = ObjectiveInfluence(model, param_infl)

gap_if = obj_infl.compute_match_influence(gap_obj, method="if")
gap_1sn = obj_infl.compute_match_influence(gap_obj, method="1sn")
```

### Action influence

```python
from clean_bt_rank import compute_influence

drop_report = compute_influence(model, gap_obj, action="drop", method="1sn")
add_report = compute_influence(
    model,
    gap_obj,
    action="add",
    method="1sn",
    candidate_mode="all_outcomes_weighted",
)
flip_report = compute_influence(model, gap_obj, action="flip", method="1sn")
```

### Iterative interventions

```python
from clean_bt_rank import make_nonpositive_condition, run_iterative_action_until_target

result = run_iterative_action_until_target(
    model,
    gap_obj,
    make_nonpositive_condition(),
    action="flip",
    influence_method="1sn",
    check_mode="refit",
)
```

## Summary

This codebase is organized around one core idea:

- fit a clean BT model once,
- expose reusable inference objects around it,
- define objectives separately,
- propagate influence through those objectives,
- then use the same machinery for reporting, interventions, and experiments.

That is why the codebase can support:

- ordinary BT ratings,
- multiple CI methods,
- uncertainty-aware objectives,
- `drop` / `add` / `flip` influence,
- iterative leaderboard manipulation,
- and Arena experiments,

without needing a different implementation for each new question.
