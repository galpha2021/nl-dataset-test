# Nonlinear model (NL) and Constrained Quadratic Model (CQM) Feature Selection using D-Wave-Compatible Models

This repository contains a Python implementation of a hybrid feature selection framework that leverages **NL** and **CQMs** to select an optimal subset of features from a dataset. It is designed for performance benchmarking to mesaure runtime and accurancy scores between both models and research in optimization-based feature selection methods.

---

## 🔍 Purpose

The goal of this project is to **compare two different optimization solvers** — a nonlinear solver and a quadratic model solver — for selecting informative and non-redundant features. It evaluates how many features to retain (`k`) from a dataset by balancing classification performance and feature redundancy via an `alpha` penalty parameter.

## ⚙️ How It Works

1. **Data Input**: The system accepts either a CSV file path or a preloaded Pandas DataFrame from openML
2. **Preprocessing**: Features and labels are cast to `float64` to ensure compatibility with solvers.
3. **Solver Choice**: Choose between:
   - `SelectFromNonlinearModel` (NL)
   - `SelectFromQuadraticModel` (CQM)
4. **Feature Selection**: The selected solver reduces the feature set to `k` features using an optimization model with a redundancy penalty `alpha` ∈ [0, 1].
5. **Evaluation**: A `RandomForestClassifier` is used with cross-validation to evaluate the accuracy of the selected feature set.
6. **Logging Results**: Results are logged in a CSV file, capturing:
   - Number of features selected
   - Execution time
   - Accuracy score
   - Selected feature indices

---

## 📊 What It Works On

The code can run on:
- Any **CSV dataset** where the target column is specified
- **OpenML datasets** via `openml-python` API

In the included example, the `Scene` dataset (OpenML ID: 312) is used for testing and benchmarking.

