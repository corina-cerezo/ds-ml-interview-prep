# Data Science & Machine Learning Interview Prep

A structured collection of Jupyter notebooks covering the core topics required for Data Scientist roles at top tech companies. Each notebook goes beyond surface-level definitions, the goal is to build genuine understanding through mathematical foundations, worked examples, real industry use cases, and interview-ready explanations.

---

## Motivation

I hold a Master's degree in Mathematical Sciences, where I studied probability theory, statistics, and machine learning at a rigorous level. After years in an applied ML engineering role, I found that knowledge I once understood deeply had become rusty from lack of use :( Honestly, that bothered me more than I expected.

This repository is my system for getting it back and keeping it. The approach is inspired by the Feynman technique: the best way to truly understand something is to explain it clearly to someone else. Every notebook is written as if I'm teaching the topic to a fellow technical professional preparing for interviews.

If you're preparing for Data Science interviews yourself, I hope this resource is useful to you too.

---

## Repository Structure

```
ds-ml-interview-prep/
│
├── 01_probability_statistics/
│   ├── 01_conditional_probability.ipynb
│   ├── 02_expectation_variance.ipynb
│   ├── 03_key_distributions.ipynb
│   ├── 04_hypothesis_testing_one_sample.ipynb
│   ├── 05_hypothesis_testing_two_samples.ipynb
│   ├── 06_binomial_proportion_tests.ipynb
│   ├── 07_bayesian_inference.ipynb
│   ├── 08_bootstrap.ipynb
│   ├── 09_monte_carlo_methods.ipynb
│   └── 10_survival_analysis.ipynb
│
├── 02_machine_learning/
│   ├── 11_linear_regression.ipynb
│   ├── 12_logistic_regression.ipynb
│   ├── 13_decision_trees.ipynb
│   ├── 14_cross_validation.ipynb
│   ├── 15_clustering.ipynb
│   ├── 16_pca.ipynb
│   ├── 17_convex_optimization.ipynb
│   └── 18_predictive_modeling_pipeline.ipynb
│
├── 03_experimentation_applied_analysis/
│   ├── 19_ab_testing.ipynb
│   ├── 20_causal_inference.ipynb
│   ├── 21_time_series_analysis.ipynb
│   └── 22_metrics_and_measurement.ipynb
│
└── README.md
```

---

## Topic List

### Category 1: Probability & Statistics
The mathematical foundation. These concepts supports everything in Categories 2 and 3.

| # | Topic | Key concepts |
|---|-------|-------------|
| 01 | Conditional probability | Bayes' theorem, independence, law of total probability |
| 02 | Expectation & Variance | $E[X]$, $Var(X)$, covariance, moment generating functions |
| 03 | Key distributions | Normal, Binomial, Poisson, t, chi-squared |
| 04 | Hypothesis testing: one sample | t-test, z-test, p-value, Type I & II errors |
| 05 | Hypothesis testing: two samples | Welch's t-test, Mann-Whitney, paired tests |
| 06 | Binomial proportion tests | z-test for proportions, chi-squared test |
| 07 | Bayesian inference | Prior, likelihood, posterior, credible intervals |
| 08 | Bootstrap | Resampling, confidence intervals, when to use it |
| 09 | Monte Carlo methods | Simulation, MCMC basics |
| 10 | Survival analysis | Kaplan-Meier, hazard function, Cox proportional hazards |

### Category 2: Machine Learning
Supervised and unsupervised models. Emphasis on mathematical intuition, not just API calls.

| # | Topic | Key concepts |
|---|-------|-------------|
| 11 | Linear regression | OLS, assumptions, $R^2$, residual analysis |
| 12 | Logistic regression | MLE, sigmoid, odds ratio, ROC/AUC |
| 13 | Decision trees | Gini impurity, entropy, pruning, Random Forest |
| 14 | Cross-validation | k-fold, LOOCV, train/val/test split |
| 15 | Clustering | K-means, hierarchical, DBSCAN, silhouette score |
| 16 | PCA | Eigendecomposition, explained variance, geometric intuition |
| 17 | Convex optimization | Gradient descent, Lagrangians, SGD, Adam |
| 18 | Predictive modeling pipeline | Feature engineering, regularization, bias-variance tradeoff |

### Category 3: Experimentation & Applied Analysis
Applying statistics to real business problems. The most differentiating category in senior DS interviews.

| # | Topic | Key concepts |
|---|-------|-------------|
| 19 | A/B testing | Experimental design, statistical power, sample size |
| 20 | Causal inference | Confounders, RCTs vs observational studies, DiD |
| 21 | Time series analysis | Stationarity, ARIMA, seasonality, forecasting |
| 22 | Metrics & measurement | Defining KPIs, metric trade-offs, guardrail metrics |

---

## Notebook Structure

Every notebook follows a consistent structure:

1. **Intuition:** What problem does this concept solve? Why does it exist?
2. **Mathematical foundation:** The theory, at the right level of depth: rigorous but not a proof seminar
3. **Assumptions:** When does this method apply, and when does it break down?
4. **Worked example:** Step-by-step solution with Python code (runnable via the Colab button)
5. **Real industry use case:** How is this actually used at tech companies?
6. **How to explain it to a non-technical stakeholder**
7. **Interview questions:** Typical questions asked at top tech companies, with answers
8. **Did you know?:** A historical note, a surprising fact, a famous counterexample, or an unexpected connection. Anything that makes the topic stick.

---

## How to Use This Repository

**If you're preparing for interviews:** Start with Category 1 in order, the later topics build on these foundations. Use the interview questions section to self-test.

**If you want to refresh a specific topic:** Each notebook is self-contained. Jump directly to what you need.

**If you want to run the code:** Click the "Open in Colab" badge at the top of any notebook. No local setup required.

---

## Progress

| # | Notebook | Status |
|---|----------|--------|
| 01 | [Conditional probability](01_probability_statistics/01_conditional_probability.ipynb) | ✅ Complete |
| 02 | Expectation & Variance | 🔜 Coming soon |
| 03 | Key distributions | 🔜 Coming soon |
| ... | ... | ... |

---

## Author

Mathematical Sciences MSc. Currently working as a Machine Learning Engineer. Preparing for Data Scientist roles at top tech companies.

*"The best way to learn is to teach."*
