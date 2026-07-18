"""
src/models.py
=============
Shared model definitions for the Online Retail customer analytics project.

This module defines the **canonical, untuned baseline classifier** used by:
  - 02_pca_lda.ipynb  (Section 4.4 feature-set comparison table)
  - 03_classification.ipynb  (starting point before cross-validated GridSearchCV tuning)

Keeping this function here (rather than copy-pasting the LogisticRegression config into
both notebooks) means that if the baseline definition ever changes, it only needs to be
updated in one place — and both notebooks automatically pick up the new version.
"""

from sklearn.linear_model import LogisticRegression


def get_baseline_classifier() -> LogisticRegression:
    """Return a fresh, untuned Logistic Regression baseline classifier.

    Configuration
    -------------
    - solver="liblinear"   : efficient for small/medium datasets; supports both l1 and l2.
    - max_iter=1000        : generous limit to ensure convergence on any reasonable dataset.
    - random_state=42      : reproducible results across runs.

    This is **Model 2 baseline** as referenced in Section 4.4 of the project spec.
    ``03_classification.ipynb`` uses this same configuration as the starting point before
    applying 5-fold GridSearchCV hyperparameter tuning.

    Returns
    -------
    LogisticRegression
        A freshly instantiated (unfitted) classifier with the canonical baseline parameters.
    """
    return LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
