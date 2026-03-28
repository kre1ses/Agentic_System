"""
Knowledge chunks for the RAG knowledge base.

Organised by agent role and task domain:
  - eda              → ExplorerAgent
  - feature_engineering → EngineerAgent
  - model_selection  → BuilderAgent
  - evaluation       → CriticAgent
  - validation       → ValidationAgent
  - domain           → all agents (rental occupancy context)
  - agent            → Planner / Coordinator
  - safety           → all agents

All chunks focus on REGRESSION (MSE optimisation on tabular data).
The competition task is rental occupancy prediction (target = days, 0-365).
"""

KAGGLE_KNOWLEDGE_CHUNKS = [

    # ── EDA for regression ──────────────────────────────────────────────
    {
        "text": (
            "For regression EDA always start with the target distribution: "
            "plot histogram and compute skewness. "
            "If skewness > 1, apply log1p transform to the target before modelling; "
            "this often reduces RMSE by 5-15% on right-skewed continuous targets. "
            "Remember to inverse-transform predictions before submission."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "regression", "target"],
    },
    {
        "text": (
            "Check for near-constant numeric features: features with std/mean < 0.01 "
            "or coefficient of variation < 1% carry almost no signal. "
            "Drop them before modelling to reduce noise and speed up training."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "regression", "feature_selection"],
    },
    {
        "text": (
            "Missing value analysis for regression: "
            "columns with > 80% NaN are almost always safe to drop. "
            "For 20-80% NaN, use median imputation for numeric and 'unknown' for categorical. "
            "Adding a binary '_was_missing' indicator for columns with > 5% NaN "
            "lets GBM models learn missingness patterns explicitly."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "regression", "missing_values", "feature_engineering"],
    },
    {
        "text": (
            "Correlation analysis for regression: "
            "compute Pearson r between every numeric feature and the target. "
            "Features with |r| > 0.7 are likely high-value; |r| < 0.01 may be noise. "
            "High inter-feature correlation (|r| > 0.95) causes multicollinearity: "
            "keep the one with higher target correlation, drop the other."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "regression", "correlation", "feature_selection"],
    },
    {
        "text": (
            "Outlier detection for regression targets: use IQR rule (Q1 - 3*IQR, Q3 + 3*IQR) "
            "or Z-score > 3. Do NOT remove outliers blindly — cap them instead (Winsorization). "
            "MSE is sensitive to outliers, so log-transforming the target often helps more "
            "than removing rows."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "regression", "outliers"],
    },
    {
        "text": (
            "For date-like columns, always extract: year, month, day_of_week, quarter, "
            "is_weekend, day_of_year, week_of_year. "
            "These cyclical features often carry strong signal for occupancy / demand tasks. "
            "Encode cyclical features with sin/cos: "
            "month_sin = sin(2π * month / 12), month_cos = cos(2π * month / 12)."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "feature_engineering", "datetime", "domain"],
    },
    {
        "text": (
            "Profile categorical features by cardinality: "
            "low (< 10 unique) → ordinal or one-hot; "
            "medium (10-50) → one-hot or target encoding; "
            "high (> 50) → target encoding or frequency encoding. "
            "Check unseen categories: categories in test not in train need a fallback "
            "strategy (e.g., map to global mean in target encoding)."
        ),
        "source": "regression_best_practices",
        "tags": ["eda", "feature_engineering", "categorical", "regression"],
    },

    # ── Feature Engineering for regression ──────────────────────────────
    {
        "text": (
            "Log-transform right-skewed numeric features (skewness > 1) with log1p. "
            "This helps linear models significantly and sometimes improves tree models too. "
            "For two-sided skew (heavy tails both sides), use Box-Cox or Yeo-Johnson transform."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "skewness", "preprocessing"],
    },
    {
        "text": (
            "Target encoding for regression: replace each category with the mean of the target "
            "for that category, computed on training data only. "
            "Use K-fold cross target encoding to prevent leakage: "
            "for each fold, compute the encoding on out-of-fold rows. "
            "Add smoothing: encoded_value = (count * category_mean + k * global_mean) / (count + k), "
            "where k ≈ 10-30 controls regularisation strength."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "categorical", "encoding", "leakage"],
    },
    {
        "text": (
            "Frequency encoding: replace each category with its count (or proportion) in the dataset. "
            "This is leakage-free, requires no target information, and captures rare vs common "
            "categories. Useful for high-cardinality IDs, ZIP codes, product codes."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "categorical", "encoding"],
    },
    {
        "text": (
            "Interaction features for regression: multiply pairs of numeric columns that "
            "have moderate correlation with the target (|r| > 0.2). "
            "Ratio features (A/B) are particularly powerful when both A and B independently "
            "predict the target. Limit to 10-20 interactions to avoid dimensionality explosion."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression"],
    },
    {
        "text": (
            "Aggregate features by group: for each categorical column C and numeric feature N, "
            "compute group statistics: mean, std, min, max, median of N grouped by C. "
            "Example: for rental data, compute mean_price_by_city, std_occupancy_by_type. "
            "Always compute these statistics on training data only to avoid leakage."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "aggregation", "leakage"],
    },
    {
        "text": (
            "Polynomial features (degree 2) for the top-5 most correlated numeric features "
            "sometimes improves linear baselines by capturing curvature. "
            "GBM models handle non-linearity natively — polynomial features rarely help them. "
            "Use PolynomialFeatures inside a sklearn Pipeline to prevent leakage."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "polynomials"],
    },
    {
        "text": (
            "Missing value imputation strategy: "
            "(1) Numeric: median is robust to outliers; mean is faster. "
            "(2) Categorical: fill with 'missing' as a new category — GBM handles it well. "
            "(3) Target-aware: KNN imputation using the k nearest rows (sklearn KNNImputer) "
            "is more accurate but 10-100x slower. Use it only if missingness is high and "
            "pattern-rich."
        ),
        "source": "regression_best_practices",
        "tags": ["feature_engineering", "regression", "missing_values"],
    },

    # ── Model Selection for regression ──────────────────────────────────
    {
        "text": (
            "For tabular regression on Kaggle, the standard winning stack is: "
            "LightGBM + XGBoost + CatBoost ensemble. "
            "LightGBM is the fastest; XGBoost is more regularised; CatBoost handles "
            "categoricals natively. Ridge regression is always a useful baseline. "
            "Start with LightGBM — it has the best default hyperparameters."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "gradient_boosting"],
    },
    {
        "text": (
            "LightGBM key hyperparameters for regression: "
            "n_estimators=1000 with early stopping on a validation set; "
            "learning_rate=0.05 (lower → better but slower); "
            "num_leaves=31-127 (higher → more expressive but overfits); "
            "min_child_samples=20 (regularisation against noisy leaves); "
            "subsample=0.8, colsample_bytree=0.8 (row/col subsampling). "
            "Use objective='regression' and metric='rmse'."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "gradient_boosting", "hyperparameters"],
    },
    {
        "text": (
            "Ridge regression is the best baseline for regression tasks. "
            "It reveals whether features are linearly predictive and is fast to train. "
            "Always StandardScale features before Ridge. "
            "If Ridge R² > 0.5, a tuned GBM will likely reach R² > 0.8. "
            "If Ridge R² < 0.2, the features need more engineering."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "baseline", "linear"],
    },
    {
        "text": (
            "Random Forest for regression: use n_estimators=300-500, "
            "max_features='sqrt' or 0.5, min_samples_leaf=5. "
            "It is more robust to outliers than GBM but typically reaches lower R². "
            "Feature importances from RF are a useful proxy for feature selection."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "random_forest"],
    },
    {
        "text": (
            "Cross-validation for regression: use KFold(n_splits=5, shuffle=True, random_state=42). "
            "Do NOT use StratifiedKFold for continuous targets. "
            "For time-series data use TimeSeriesSplit to prevent future leakage. "
            "Report both CV mean and std: high std (> 20% of mean) signals overfitting or "
            "an unstable model."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "evaluation", "regression", "cross_validation"],
    },
    {
        "text": (
            "Ensembling regression models: average predictions from 3-5 diverse models "
            "(Ridge + RF + LightGBM) reduces variance and typically improves RMSE by 2-5%. "
            "Stacking: train a meta-learner (Ridge) on out-of-fold predictions as features. "
            "Weight models by 1/MSE_cv for a simple weighted average."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "ensembling"],
    },
    {
        "text": (
            "Hyperparameter tuning order for GBM regression: "
            "(1) Fix n_estimators high (1000) with early stopping. "
            "(2) Tune max_depth / num_leaves for model complexity. "
            "(3) Tune subsample and colsample_bytree for regularisation. "
            "(4) Lower learning_rate to 0.01-0.05 and increase n_estimators proportionally. "
            "Optuna with TPE sampler converges faster than RandomSearch."
        ),
        "source": "regression_best_practices",
        "tags": ["model_selection", "regression", "hyperparameters", "tuning"],
    },

    # ── Evaluation metrics for regression ───────────────────────────────
    {
        "text": (
            "Primary metric for regression: MSE (Mean Squared Error). "
            "RMSE (sqrt of MSE) is in the same units as the target and easier to interpret. "
            "MAE (Mean Absolute Error) is more robust to outliers. "
            "R² (coefficient of determination) measures proportion of variance explained: "
            "R² = 1 means perfect fit; R² = 0 means the model predicts the mean. "
            "Negative R² means the model is worse than predicting the mean."
        ),
        "source": "regression_best_practices",
        "tags": ["evaluation", "regression", "metrics"],
    },
    {
        "text": (
            "If the target is skewed and you apply log1p transform, "
            "evaluate on log-scale (RMSLE = RMSE of log1p predictions) during training, "
            "but always report RMSE on the original scale for interpretability. "
            "Submit predictions in the original scale (apply expm1 to inverse-transform)."
        ),
        "source": "regression_best_practices",
        "tags": ["evaluation", "regression", "metrics", "target"],
    },
    {
        "text": (
            "Residual analysis after training: plot (predicted - actual) vs predicted. "
            "Patterns in residuals (funnel shape, curved trend) reveal model deficiencies. "
            "Heteroscedasticity (variance grows with predicted value) often means "
            "a log-transform of the target or a Tweedie/Poisson objective is needed."
        ),
        "source": "regression_best_practices",
        "tags": ["evaluation", "regression"],
    },
    {
        "text": (
            "Feature importance interpretation: "
            "GBM split importance counts how often a feature is used in splits. "
            "Permutation importance (sklearn) measures MSE increase when a feature is shuffled — "
            "more reliable but 2x slower. "
            "SHAP values give per-sample importance and are the gold standard. "
            "Features with near-zero importance can usually be dropped without MSE regression."
        ),
        "source": "regression_best_practices",
        "tags": ["evaluation", "regression", "feature_selection"],
    },

    # ── Validation & leakage ─────────────────────────────────────────────
    {
        "text": (
            "Data leakage is the #1 source of over-optimistic regression models. "
            "Fit ALL transformers (scalers, encoders, imputers) ONLY on training data, "
            "then apply to test. Use sklearn Pipeline to enforce this mechanically. "
            "Columns highly correlated with target (|r| > 0.95) are leakage suspects: "
            "verify they are legitimately available at prediction time."
        ),
        "source": "regression_best_practices",
        "tags": ["validation", "leakage", "regression", "pipeline"],
    },
    {
        "text": (
            "ID columns (unique or near-unique identifiers) must be dropped before training. "
            "They have no predictive power but can cause severe overfitting in tree models "
            "by memorising row-identity. Detect ID columns by: uniqueness ratio > 0.9, "
            "column name containing 'id', 'index', 'key', 'row', 'record'."
        ),
        "source": "regression_best_practices",
        "tags": ["validation", "leakage", "feature_engineering", "regression"],
    },
    {
        "text": (
            "Train-test schema validation before pipeline start: "
            "assert that all feature columns in train exist in test. "
            "Check that the target column is absent from test. "
            "Verify that column dtypes are consistent (object in train → object in test). "
            "Schema mismatches caught early prevent cryptic errors downstream."
        ),
        "source": "regression_best_practices",
        "tags": ["validation", "regression"],
    },
    {
        "text": (
            "Target leakage detection heuristics for regression: "
            "(1) Name-based: columns named 'target', 'label', 'price_final', 'outcome'. "
            "(2) Correlation-based: |Pearson r| > 0.9 with target is suspicious. "
            "(3) Rank-based: Spearman ρ > 0.9 catches monotone non-linear leakage. "
            "Always manually inspect the top-5 most correlated features."
        ),
        "source": "regression_best_practices",
        "tags": ["validation", "leakage", "regression"],
    },

    # ── Domain knowledge: rental occupancy prediction ────────────────────
    {
        "text": (
            "Rental occupancy prediction: the target is typically days occupied per period (0-365). "
            "Key predictors: listing type, location, price tier, number of bedrooms/bathrooms, "
            "review score, host response rate, amenities count. "
            "Seasonal effects are strong — always extract month and quarter from date columns."
        ),
        "source": "domain_rental_occupancy",
        "tags": ["domain", "regression", "rental"],
    },
    {
        "text": (
            "Seasonality in rental demand: occupancy peaks in summer (Jun-Aug) and "
            "around holidays. Winter months typically see 20-40% lower occupancy. "
            "Day-of-week matters: weekends command higher occupancy for leisure rentals. "
            "Feature: is_peak_season = (month in {6,7,8}) or is_holiday_week."
        ),
        "source": "domain_rental_occupancy",
        "tags": ["domain", "regression", "rental", "datetime", "feature_engineering"],
    },
    {
        "text": (
            "Price vs occupancy: there is typically an inverted-U relationship — "
            "very low and very high prices both correlate with lower occupancy. "
            "Price percentile within the same listing_type and location is more informative "
            "than raw price. Consider creating: price_rank_by_location feature."
        ),
        "source": "domain_rental_occupancy",
        "tags": ["domain", "regression", "rental", "feature_engineering"],
    },
    {
        "text": (
            "Review metrics for rental occupancy: "
            "review_score and number_of_reviews are strong occupancy predictors. "
            "High review count + high score → consistently booked. "
            "Low review count → new listing, behaviour is harder to predict. "
            "Feature: log1p(number_of_reviews) reduces the effect of extreme reviewers."
        ),
        "source": "domain_rental_occupancy",
        "tags": ["domain", "regression", "rental", "feature_engineering"],
    },
    {
        "text": (
            "Location features for rental: city, neighbourhood, latitude/longitude. "
            "Cluster listings by lat/lon using KMeans or BinLatLon to create "
            "a 'micro-market' feature. Mean target by neighbourhood is a powerful "
            "target encoding — apply cross-fold to avoid leakage."
        ),
        "source": "domain_rental_occupancy",
        "tags": ["domain", "regression", "rental", "feature_engineering", "encoding"],
    },

    # ── Agent architecture (kept from original) ──────────────────────────
    {
        "text": (
            "ReAct agent pattern: at each step the agent (1) Reasons about the current "
            "state, (2) Acts by calling a tool, and (3) Observes the result before "
            "deciding the next action. This prevents hallucinated tool calls and grounds "
            "the agent's actions in observed data."
        ),
        "source": "agent_architecture",
        "tags": ["agent", "react", "architecture"],
    },
    {
        "text": (
            "Planner-Executor-Critic loop: "
            "Planner decomposes the goal into ordered subtasks. "
            "Executor performs each subtask using tools. "
            "Critic evaluates output and provides corrective feedback. "
            "Repeat until quality threshold is reached or MAX_CRITIQUE_ROUNDS exceeded."
        ),
        "source": "agent_architecture",
        "tags": ["agent", "planning", "critic"],
    },
    {
        "text": (
            "Validation agent as Phase 0: run input validation before any LLM calls. "
            "Check file existence, train/test schema alignment, target correctness, "
            "ID-column detection, and leakage heuristics. "
            "Return a structured validation_report consumed by Engineer and Builder. "
            "Stop the pipeline immediately (fail-fast) on critical errors."
        ),
        "source": "agent_architecture",
        "tags": ["agent", "validation", "architecture"],
    },
    {
        "text": (
            "RAG (Retrieval-Augmented Generation) in multi-agent systems: "
            "retrieve domain knowledge relevant to each agent's current task "
            "and inject it into the system prompt. "
            "Use tag-filtered retrieval to give each agent role-specific chunks: "
            "EDA agent → 'eda' tag, Engineer → 'feature_engineering', Builder → 'model_selection'."
        ),
        "source": "agent_architecture",
        "tags": ["agent", "rag", "architecture"],
    },
    {
        "text": (
            "Experiment memory as RAG: after each run, persist model results and "
            "feature decisions as knowledge chunks. Future runs retrieve these to "
            "avoid repeating failed experiments and to warm-start with the best "
            "hyperparameters from prior runs."
        ),
        "source": "agent_architecture",
        "tags": ["agent", "memory", "rag", "architecture"],
    },

    # ── Safety ────────────────────────────────────────────────────────────
    {
        "text": (
            "Prompt injection defence: never interpolate unvalidated user text directly "
            "into system prompts. Use a separate 'data' field and instruct the model "
            "to treat it as data, not instructions. "
            "Validate all file paths and column names before embedding them in prompts."
        ),
        "source": "agent_security",
        "tags": ["safety", "prompt_injection"],
    },
    {
        "text": (
            "Sandboxed code execution: run LLM-generated code in a subprocess with "
            "a timeout and restricted imports (no socket, no os.system, no subprocess). "
            "Validate generated code with AST analysis before execution. "
            "Never execute code with eval() directly."
        ),
        "source": "agent_security",
        "tags": ["safety", "sandboxing", "code_execution"],
    },
]
