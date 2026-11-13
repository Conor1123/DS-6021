# DS-6021 Quiz 4 — review sheet (11-13)

Scope: Missing Data / Multinomial Logistic / KNN / K-means. Paper, closed book, no AI, no calculator.

## 1. Missing data

- MCAR: no pattern; missingness unrelated to every variable, observed or missing. Rare, statistically best.
- MAR: missingness depends on other OBSERVED variables, not on the missing value. "Observed" includes the outcome.
- MNAR: the reason it is missing is the value itself (scale fails above 250 lbs; high earners skip income).
- Test to apply: is the driver of missingness a column that is IN the data? Yes -> MAR. No (it is the missing value) -> MNAR.
- Checkability: MAR is diagnosable (model missingness on the recorded columns, compare groups). MCAR can only be falsified. MNAR is unverifiable from the data - needs knowledge of the mechanism.
- Options: drop rows (loses that row's OTHER observed values too; only unbiased under MCAR; smaller n -> wider CIs), drop columns (only when the non-missing values are not meaningful), add an "is missing" indicator column (deck flags it as controversial), impute.
- Encode deliberately: 0 / N/A / NaN can each mean missing, a real value, or not applicable; -1 / 99 / 999 may be proxies for missing.
- Mean or median for numeric, mode for categorical. Also KNN imputation, regression imputation, MICE (several imputed datasets, pooled).
- Mean imputation: spread collapses (all filled values identical, SD falls) AND the reported SE is too SMALL, because imputed rows are counted as measured data - n rises while information does not. Small SE is not evidence of a reliable coefficient.
- Regression imputation: fit the missing variable on the others, predict. Add residual noise (stochastic) to restore scatter; deterministic imputation puts every filled value exactly on the line -> no variance, the variable looks like an exact function of its predictors, so its own coefficient is pulled toward 0 with a suspiciously small SE.
- Changing the noise seed moves the estimate: single imputation ignores imputation uncertainty.
- Impact on predictions: a non-MCAR drop selects the sample, and whatever bias is in the fit carries into every prediction; mean-imputed rows predict toward the average because the feature carries no information there.

## 2. Multinomial logistic regression

- Binary: one coefficient set, one score z, p = 1/(1+e^-z) = e^z/(1+e^z). logit(p) = ln(p/(1-p)).
- Multinomial: ONE joint model, a separate coefficient set PER CLASS (K rows in coef_), no reference class.
- Softmax: p_k = exp(z_k) / sum_j exp(z_j). Exponentiate, then divide by the sum; positive and sums to 1 by construction.
- One-vs-rest: K independent binary models, each "class k vs the rest"; probabilities need not sum to 1; pick the largest. One-vs-one: K(K-1)/2 pairwise models plus a vote. With K=2 all reduce to ordinary logistic.
- sklearn defaults: multinomial (softmax) for 3+ classes, and an L2 (ridge) penalty always on at C=1.0. C is the INVERSE of penalty strength; C=np.inf is the unpenalized model. l1_ratio=1 lasso, 0 ridge.
- Coefficient meaning: a one-unit rise in x adds b to class k's score (its log-odds), so the odds multiply by e^b. It is NOT a change in probability; the probability change depends on the starting probability, and softmax pushes the other classes down.
- If predictors were standardized, the unit is one standard deviation, and the penalty depends on scale - that is why you scale.
- Only row DIFFERENCES are interpretable: odds of class k vs class k' multiply by exp(b_k - b_k') per one unit. A single row quoted as an odds ratio is the classic error.
- Lasso pattern: a feature nonzero in one row and zero in the others is used to separate that class from the rest; all rows zero means the feature is dropped.
- Baselines: predicting 1/K for all classes scores log loss = ln K (ln 3 = 1.10); with unbalanced classes the base-rate baseline is lower than ln K.
- Multiclass confusion matrix: rows actual, columns predicted; each class gets its own precision and recall; summarize macro (each class equal) or weighted (by class size).

## 3. KNN

- Non-parametric, no functional form, no coefficients, no real training phase: the data are stored and prediction does the work. k is a hyperparameter chosen by cross-validation, not estimated from data.
- Classification: majority vote among the k nearest neighbors; predicted probability = share of those k in the class. Regression: average of the k neighbors' y values.
- Distances for numeric: Euclidean (default, p=2), Manhattan (p=1, sum of absolute differences, better at very high dimension), Minkowski general form.
- Categorical: no natural distance - coding 1/2/3 invents an order and a spacing. Use Hamming (count of positions that differ; sklearn's metric="hamming" returns the SHARE that differ) or Jaccard (mismatches as a share of features present in either; distance = 1 - index).
- Hamming is between two rows, not per row. Write out position by position and count MISMATCHES.
- Scaling: the distance sums each feature's contribution, so the large-unit feature dominates. Standardize to mean 0, SD 1; fit the scaler on the training data only.
- k small (k=1): low bias, high variance, boundary follows every point, every probability is 0 or 1, so one mistake costs enormous log loss. k large: high bias, low variance, smoother boundary. k=n predicts the overall mean (or the majority class).
- Odd k in binary classification avoids tied votes.
- Compute: brute force, KDTree, BallTree; sklearn defaults to algorithm="auto".
- KNN vs logistic: KNN's boundary can take any shape and weights all features equally; logistic is linear in the features and can drop them via the penalty. Similar performance suggests a linear boundary is adequate.

## 4. K-means

- Supervised: features AND a labeled response, predictions checkable. Unsupervised: features only, no response, no right answer to score against, so evaluation is harder.
- Objective: partition into K clusters minimizing WCSS (within-cluster sum of squares); each point in exactly one cluster; the centroid is the mean of its members.
- Lloyd's algorithm: initialize (random centers or random assignment) -> update (recompute each centroid as the mean of its members) -> assign (move every point to the nearest centroid) -> repeat until no assignment changes.
- It must stop: the update step can only lower WCSS (the mean minimizes squared distance) and the assign step can only lower each point's distance; WCSS never increases over finitely many partitions.
- It stops at a LOCAL optimum that depends on the start, so two runs can give different clusterings (class: 379.4 vs ~486). Trying every partition is impossible.
- Fixes: n_init restarts, keeping the run with the lowest inertia_ (with k-means++ the default is a single run); k-means++ picks the first center uniformly at random and each next center with probability proportional to D(x)^2, the squared distance to the nearest center already chosen, so the starts spread out.
- Scaling matters more than anywhere else: grams vs millimetres turns the clusters into bands of the large-unit variable.
- Choosing K: WCSS always falls as K grows (K=n gives 0), so it cannot select. Elbow = take the bend where returns diminish. Silhouette = a(i) mean distance to the OTHER POINTS in its own cluster, b(i) mean distance to the points in the nearest other cluster, s = (b-a)/max(a,b); s near 1 well clustered, near 0 on a boundary, negative means probably the wrong cluster; average over points and take the largest - and unlike WCSS it IS comparable across K. Gap statistic = compare log(W_K) against reference data with no clusters; take the smallest K where the gap stops growing. Methods disagree; check stability and usefulness.
- Categorical data: number codes invent order; one-hot makes centroids into proportions (a center that is 63% Dream and 90% male is not a real observation); the mean is the right center only for squared Euclidean - for Hamming the right center is the mode (K-modes), mixed data use K-prototypes, and K-medoids or hierarchical clustering need no mean at all.
- K-means favors round, similarly sized clusters; outliers pull centroids and every point is forced into some cluster.

## Traps you personally hit while drilling

1. Calling dependence on a recorded outcome "MNAR" - if the driver is in the data it is MAR.
2. Saying mean imputation increases the SE - the reported SE is understated.
3. Looking for a reference class in a multinomial model - there is none; difference the rows.
4. Saying odds "double the chance" - odds multiply, probabilities do not.
5. Writing z for p (score versus probability).
6. Silhouette described with centroids - it uses point-to-point distances.
7. Elbow described as "the highest K with diminishing returns" - you take the bend.
8. Hamming reported per row, or counting matches instead of mismatches.
9. KNN classification described as "the closest class" - it is a majority vote of the k neighbors.
10. Softmax given as the decision rule - argmax picks the class; softmax only produces the numbers.

## Formulas

- logit(p) = ln(p/(1-p)) = b0 + b1 x1 + ...; p = e^z/(1+e^z) = 1/(1+e^-z)
- softmax: p_k = e^{z_k} / sum_j e^{z_j}
- odds ratio for class k vs k' per one unit of x: exp(b_k - b_k')
- log loss (binary): -(1/n) sum [ y ln(p) + (1-y) ln(1-y) ]; per row = -ln(p on the true class); K-class chance baseline = ln K
- Euclidean: sqrt(sum (x0j - xj)^2); Manhattan: sum |x0j - xj|; Minkowski: (sum |x0j - xj|^q)^(1/q)
- Hamming = count of differing positions (sklearn: that count / number of positions); Jaccard distance = 1 - |A n B|/|A u B|
- standardize: z = (x - mean)/SD
- WCSS = sum_k sum_{i in C_k} ||x_i - mu_k||^2
- silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
- OLS SE of a slope falls as sqrt(Sxx) rises; imputation inflates n without inflating Sxx -> too-small SE
