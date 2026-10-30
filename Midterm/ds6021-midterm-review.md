# DS-6021 Midterm — Review Sheet

**Format:** eight questions, fifty minutes, roughly six minutes per question, several parts per question. Paper,
closed book, no reference material. One to three sentences per part, bullets fine. Points are awarded **per
clause** — always state the verdict, not just the mechanism.

---

# Part 1 — The ones I got outright wrong. Fix these first.

## 1.1 Logit, odds, probability

Three lines, nothing else:

- `logit(p) = ln( p / (1 − p) )` — the log of the odds.
- `odds = e^z = p / (1 − p)`
- `p = odds / (1 + odds)` — **plus sign, always**

Anchors:
- `logit(0.5) = 0`, because the odds are 1 and ln 1 = 0.
- Any answer must satisfy 0 < p < 1.
- `p = 1/(1 + e^−z)` is the same thing, but only use it if the sign is nailed down.

The trap that cost five attempts: **e^z and e^(−z) are reciprocals.** If z is negative then e^z is below 1 while
e^(−z) is above 1, and using one in place of the other puts the answer on the wrong side of 0.5. Pick one route —
`odds = e^z`, then `p = odds/(1 + odds)` — and never invert inside an exponent.

Sanity check that catches the error every time: a negative z must give p below 0.5, a positive z must give p above
it. A probability outside (0, 1) means the substitution was wrong.

## 1.2 Log loss

`LL = −(1/n) Σ [ y·ln p̂ + (1 − y)·ln(1 − p̂) ]`

- The **leading minus is not optional.** ln p̂ is negative for any probability, so the minus is what makes the score
  positive. A log loss without it comes out negative, which is impossible.
- Branch: y = 1 costs `−ln p̂`; y = 0 costs `−ln(1 − p̂)`.
- **Unbounded:** p̂ → 0 on a positive row costs ∞; p̂ → 1 on a negative row costs ∞. One confident mistake can
  outweigh thousands of easy correct calls.
- `exp(−LL)` = geometric mean of the probability given to the true label — a probability-scale summary, **not**
  accuracy.
- Two references to name, neither needing a memorised value: the **base-rate null model** (predict the class rate
  for everyone; this is the bar a real model must beat) and **ln 2**, the log loss of predicting 0.5 for everyone,
  i.e. a coin flip.
- ln 2 is the "knows nothing" line, **not a ceiling**. Log loss has no upper limit, so any score above it is worse
  than random guessing.
- Killer example, which states the whole argument: a model assigning just above 0.5 to every row has perfect
  accuracy at the 0.5 threshold, while its log loss lands at about the coin flip's — perfect labels, zero
  information.

## 1.3 The three components of expected test error

`expected test error = bias² + variance + irreducible error`

- **bias²** — how far the *average* prediction sits from the truth. Falls as flexibility rises.
- **variance** — how much the *fitted model* moves when the training sample changes. Rises with flexibility.
- **irreducible** — noise in y. No model removes it.

Do not call the noise "variance in the data". Variance is a property of the fitted model; the noise floor is a
separate third term.

Signatures:
- big gap, low training error → **variance** (overfitting)
- small gap, both errors high → **bias** (underfitting). "High" needs a reference: a more flexible model on the same
  data, or the irreducible noise floor.
- training error falls monotonically with flexibility; **test error is U-shaped**.

## 1.4 ColumnTransformer vs Pipeline

- **ColumnTransformer** = column-wise *routing*. Different transformers applied to different column groups
  (num → scaler, cat → encoder); nothing else passes through unless `remainder` says so.
- **Pipeline** = sequential *steps*, each a transformer or a model, the whole chain behaving as one estimator. The
  model is fit because the **last step is a model** — not because "Pipeline fits the regression".
- During CV the entire chain (scaler, encoder, model) is refit on each fold's training rows only.

## 1.5 Odds phrasing

Say: "the **odds** are `100(e^β − 1)` percent lower (or higher)", or "`e^β` times the **odds**".

Never: "`e^β` times as likely" or "N percent less likely". Those are probability claims, and the coefficient gives
you no probability change. The same multiplicative change in odds corresponds to a **different** probability change
at every starting probability, so any "times as likely" or "percent less likely" sentence is wrong even when the
odds arithmetic is right.

---

# Part 2 — Topic notes

## 2.1 Bias-variance and model complexity

- More flexibility → bias down, variance up. Training error always falls; test error is U-shaped.
- Ordering: linear regression = highest bias, lowest variance. Within a polynomial family, degree sets flexibility;
  within a spline family, the knot count does. More of either = higher variance.
- Polynomial = one global fit; a high-degree term `x^d` dominates outside the data range, so the curve runs away past
  the last observation, and one changed point moves the whole fit.
- Splines = local pieces between knots, more stable at the edges; a **natural cubic spline is linear beyond the
  boundary knots**, which is what stops it running away.
- Polynomials and splines are still *linear models*: linear in the **coefficients**. `x²`, `x³` and spline basis
  terms are engineered features computed from x.
- Choosing: the **simplest model within about one standard error of the best** CV error. Not "because compute is
  cheaper" — that is the wrong criterion.

## 2.2 Linear regression

- OLS minimises `RSS = Σ(y − ŷ)²`. Procedure, in words: write RSS, differentiate with respect to **each
  coefficient**, set each derivative to 0, solve the equations simultaneously (one equation per coefficient).
- Simple regression closed forms: `b1 = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²`, `b0 = ȳ − b1·x̄`.
- Interpretation, all three clauses: per **one unit** of x, y changes by b1 **on average**, **holding the other
  predictors fixed**; the intercept is the predicted y at x = 0.
- Units convert in both directions: coefficient units = (units of y)/(unit of x). Always convert the prediction back
  into the units of y — a prediction is not finished until it is in the response's real units.
- Coefficient table: `t = coef/SE`, 95% CI ≈ `coef ± 2·SE`. A CI crossing zero ⇔ not significant. Large SE → wide CI
  → small t → large p → no evidence of an effect. State the verdict ("not significant"), not just the size.
- `R² = 1 − RSS/TSS`, `TSS = Σ(y − ȳ)²`. Share of variation **in y** explained by the model.
- R² can only stay the same or rise when a predictor is added: the larger model can always set the new coefficient to
  zero, so RSS cannot increase.
- `adj R² = 1 − (1−R²)(n−1)/(n−p−1)`: charges for extra predictors, so it can fall while R² rises → don't add.
- R² is in-sample and unpenalised: it says nothing about out-of-sample prediction and nothing about causation. A
  small R² can sit alongside a tiny p-value, so a low R² is not evidence of no effect.
- **VIF** `= 1/(1 − R²_j)`, where R²_j comes from regressing X_j on **all the other predictors** (not on y).
- Rules of thumb: 1 unrelated, above 5 worth a look, above 10 likely serious. VIF is a variance multiplier: the
  coefficient's variance is inflated by the VIF and its standard error by the square root of the VIF.
- VIF inflates SEs, widens CIs, raises p-values and makes coefficients uninterpretable. It does **not** damage fitted
  values or predictions: adding a near-duplicate column can leave R² almost untouched while the SE of the original
  coefficient multiplies.
- Drop or not depends on the goal: **inference** → drop one, combine, or regularize. **Prediction** → let CV decide.
  VIF never tells you *which* predictor it is redundant with; check the correlations for the partner.

## 2.3 Cross-validation

- Each fold's error is computed on rows that fold's model never trained on, so the average estimates expected
  **out-of-sample** error — of the modelling approach, not of the single model you finally fit on all rows.
- Better than one split because a single split depends on which rows happened to land in test; the spread across
  folds is information too.
- **Bias by k** (the graded mechanism): each fold model trains on `(k−1)/k` of the rows, while the model you ship
  trains on all of them. Less training data ⇒ worse model ⇒ every fold's error is systematically higher, so the
  estimate is pessimistic. That bias falls as k rises; LOOCV, at `(n−1)/n`, is nearly unbiased.
- **Variance of the estimate** rises with k: the training sets overlap more, so fold errors are correlated and the
  average is noisier. LOOCV's estimate is the noisiest and its cost is n fits. What you gain in bias you pay for in
  variance and compute, which is why a moderate k is the usual pick.
- Test-fold size affects noise, not bias.
- In-sample vs CV: the in-sample number is optimistic, because the model has seen those rows. CV being higher is
  expected and is the honest estimate. Coefficients that agree across folds within their 95% CIs mean the model is
  stable and not fitting noise.
- `StratifiedKFold` for classification, so each fold keeps the class rate.

## 2.4 Categorical variables and interactions

- k levels → **k − 1** dummies plus an intercept. The omitted level is the **reference level**; each coefficient is
  the average difference in y versus the reference, holding the others fixed.
- All k dummies **plus** an intercept is perfectly collinear (the dummies sum to the intercept column): rank
  deficient, no unique solution, coefficients uninterpretable. sklearn does not error — it silently returns a
  minimum-norm answer.
- Changing the reference level changes the coefficients and the intercept; fitted values, predictions and R² stay
  identical.
- A categorical stored as numeric codes imposes an order and equal spacing between neighbouring levels. Encode as
  dummies instead; the extra flexibility is then charged for by adjusted R², so an adjusted R² that still improves
  after the penalty means the encoding earned its place.
- Interaction: `Y = β0 + β1X + β2D + β3(X·D)`
  - D = 0: `β0 + β1X`; D = 1: `(β0 + β2) + (β1 + β3)X`
  - β1 = slope for the **reference group only**, not an overall effect
  - β2 = difference in intercepts at X = 0
  - β3 = difference in slopes, group 1 minus group 0
  - The gap between groups at value X is `β2 + β3·X`: equal to β2 only at X = 0, and constant across X only if
    β3 = 0.
- Keep both main effects whenever the interaction is in the model, otherwise the interaction column absorbs the main
  effect and β3 stops meaning "the extra effect".

## 2.5 Logistic regression

- `logit(p) = ln(p/(1−p)) = β0 + β1x1 + …`; inverse `p = 1/(1+e^−z)`; `odds = e^z`; `p = odds/(1+odds)`.
- Coefficients add on the log-odds scale and multiply the odds: β → `OR = e^β`. An OR near 1 ⇒ no clear effect.
- Interpret as odds, never as probability: "`e^β` times the odds", not "`e^β` times as likely".
- Intercept = log-odds for the reference group (numeric predictors at 0, or at their means if scaled). Convert with
  odds then p.
- Log loss is the scoring rule (Part 1.2). Baselines: the base-rate null model, and ln 2 for the coin flip.
- **Accuracy is a bad metric** when classes are imbalanced: it thresholds at 0.5, ignores the probabilities, and a
  "never predict the rare class" model scores the base rate as its accuracy while catching nothing (recall 0).
- Confusion matrix `[[TN, FP], [FN, TP]]`:
  - precision = `TP/(TP+FP)`: of the rows you flagged, how many were real
  - recall = `TP/(TP+FN)`: of the real rows, how many you caught
  - F1 = `2PR/(P+R)`: harmonic mean, high only if both are; always within [0, 1]
  - accuracy = `(TP+TN)/N`
- AUC = area under the ROC curve (recall against false-positive rate `FP/(FP+TN)`) as the threshold sweeps; it equals
  the probability that a random positive is scored above a random negative. Threshold-free, because the curve *is*
  every threshold.
- Lowering the threshold flags more rows: TP ↑, FP ↑ → recall ↑, precision ↓. Screening with a costly miss →
  optimise **recall** and lower the threshold. Precision first only where a false alarm is the expensive error.

## 2.6 Regularization

- Minimise `RSS + λ·penalty`. Ridge: `λ Σ β²`. Lasso: `λ Σ |β|`. Elastic net mixes the two, with `l1_ratio` setting
  the mix.
- Effect: coefficients shrink toward 0 → training error up a little, **bias up, variance down**, more stable
  estimates, better generalisation when λ is chosen by CV.
- **Ridge** shrinks toward 0 but never exactly to 0, so every feature stays; over a correlated group it **spreads**
  the weight.
- **Lasso** sets some coefficients exactly to 0 → automatic feature selection; over a correlated group it tends to
  **keep one** and zero the rest, and which one it keeps is unstable.
- **Standardisation is required** for both because the penalty acts on coefficient magnitude: without it, features on
  different scales are penalised unequally and λ is meaningless. OLS needs no scaling — rescaling changes only the
  units and the fit is identical.
- Lasso vs forward/backward selection: no k to choose (CV picks the penalty), all features considered at once rather
  than greedily, and it shrinks as well as selects. Limits: correlated-group instability, biased coefficients
  (relaxed lasso refits OLS on the survivors), dependence on scaling and the alpha grid.

## 2.7 sklearn pipelines

- Stages: impute/transform numerics → scale → one-hot encode categoricals → model. `Pipeline` chains the steps;
  `ColumnTransformer` routes column groups to their own transformers.
- Encoders: `StandardScaler` (mean 0, SD 1 — one unit = one SD), `MinMaxScaler` (to [0, 1] — one unit = the original
  range), `RobustScaler` (median/IQR — one unit = one IQR), `OneHotEncoder(drop="first")` (k − 1 columns, first level
  dropped as the reference).
- `fit` **learns** the parameters (mean, SD, categories); `transform` **applies** the stored ones. That is why the
  test rows never influence them.
- Leakage: scaling or encoding on the full dataset before splitting lets the held-out rows into those parameters, so
  the reported score is **too optimistic**. Correct order: split first, then fit the pipeline on the training rows;
  inside CV the whole chain refits per fold.
- Leakage (the transformation saw held-out rows) is distinct from in-sample optimism (the model saw the rows it is
  scored on). Both inflate the score; only the first is leakage.

---

# Part 3 — Formulas

**Linear regression**
- `RSS = Σ(y − ŷ)²`
- `b1 = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²`, `b0 = ȳ − b1·x̄`
- `R² = 1 − RSS/TSS`, `TSS = Σ(y−ȳ)²`; simple regression: `R² = r²`
- `adj R² = 1 − (1−R²)(n−1)/(n−p−1)`
- `t = coef / SE`; 95% CI ≈ `coef ± 2·SE`
- `VIF = 1/(1 − R²_j)`, R²_j from regressing X_j on the other predictors
- `ŷ = b0 + b1x1 + b2x2 + …`; coefficient units = (units of y)/(unit of x)

**Bias-variance**
- `expected test error = bias² + variance + irreducible error`
- `bias = E(ŷ) − f(x)`; `variance = Var(ŷ)`

**Polynomials and splines**
- Degree d polynomial: `β0 + β1x + β2x² + … + βd·x^d` (d + 1 parameters)
- Linear spline: `β0 + β1x + β2(x−c1)+ + β3(x−c2)+`, slopes `β1`, `β1+β2`, `β1+β2+β3`
- Cubic spline basis: `x, x², x³, (x−c1)+³, (x−c2)+³`; natural cubic = linear beyond the boundary knots

**Interactions**
- `Y = β0 + β1X + β2D + β3(X·D)`
- `D = 0: Ŷ = β0 + β1X`; `D = 1: Ŷ = (β0+β2) + (β1+β3)X`
- Group gap at X: `β2 + β3·X`

**Logistic regression**
- `logit(p) = ln( p/(1−p) ) = β0 + β1x1 + …`
- `odds = e^z`, `p = odds/(1+odds)`, `p = 1/(1+e^−z)`
- `OR = e^β`; percent change in odds = `100·(e^β − 1)`

**Classification metrics**
- `accuracy = (TP+TN)/N`
- `precision = TP/(TP+FP)`
- `recall = TP/(TP+FN)`
- `F1 = 2PR/(P+R)`
- false positive rate = `FP/(FP+TN)`
- `log loss = −(1/n) Σ [ y·ln p̂ + (1−y)·ln(1−p̂) ]`
- `exp(−log loss)` = geometric mean of the probability given to the true label

**Regularization**
- Ridge: minimise `RSS + λ·Σβ²`
- Lasso: minimise `RSS + λ·Σ|β|`
- Elastic net: both penalties, mix set by `l1_ratio`

**Cross-validation**
- Each fold trains on `(k−1)/k` of the rows; the score is the mean of the k fold errors
