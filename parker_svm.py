#!/usr/bin/env python
from textwrap import dedent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Dark theme for plots ᕙ(▀̿ĺ̯▀̿ ̿)ᕗ
plt.style.use("dark_background")  

# Load data
def read_file(path):
    """
    reads file depending on file extension.
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.

    OUTPUT:
        dframe: (pd.DataFrame) Dataframe of the dataset
    """
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format!\nMust be a string")

    try:
        if path.endswith("csv") or path.endswith("txt"):
            dframe = pd.read_csv(path)

        elif path.endswith("dat"):
            dframe = pd.read_csv(path, sep=r"\s+")

    except Exception as err:
        print(f"OOPZ!\n{err}")

    return dframe

def split_feats_target(dataframe, target):
    """
    Split data into predictors and binary target
    """
    # Copy for safety
    X = dataframe.drop(columns=[target]).copy()
    # Encode target values for SVM
    y = dataframe[target].astype(int)

    return X, y

def get_feature_types(X):
    """
    Identify numeric/categorical/boolean features.
    ------------------------------------------
    INPUT:
        X: (pd.DataFrame) Predictor matrix

    OUPTUT: (tuple)
        numeric_feats: (list of str) Numerical columns
        categorical_feats: (list of str) Categorical and boolean columns
    """
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    return numeric, categorical

def the_preprocessor(numeric, categorical):
    """
    Build preprocessing transformer for SVM, where numeric features are normalized z = (z - mu) / sig.
    Categorical/boolean features are one-hot encode.
    -------------------------------------------------
    INPUT:
        numeric: (list of str) Numeric data
        categorical: (list of str) Categorical data

    OUTPUT:
        ColumnTransformer (preprocessing transformer)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
                categorical)
        ]
    )

def build_pipeline(preprocess):
    """
    BUilds full preprocessing + SVM pipeline, where I used
    class_weight='balanced' since the target is severly imbalanced.
    Probability=True so the ROC-AUC can be computed from the predicted probabilities.
    max_iter=1000 for avoiding long convergenence limits.
    ------------------------------------------------
    INPUT:
        preprocess: (ColumnTransformer) Preprocessing step

    OUTPUT:
        (Pipeline) Sklearn pipeline
    """
    return Pipeline(
        [
            ("prep", preprocess),
            "svm",
            SVC(
                class_weight="balanced",
                probability=True,
                max_iter=1000
            )
        ]
    )

def define_param_grid():
    """
    Defines a reduced hyperparameter grid.
    
    Linear Kernel: c ∈ {0.1, 1, 10}

    RBF Kernel: c ∈ {0.1, 1, 10}
    gamma ∈ {'scale', 0.01}

    Trims runtime while allowing a real comparison between linear and nonlinear hyperplanes
    ------------------------------------------------------------
    iNPUT:
        None

    OUTPUT:
        (list of dict) Parameter grid for GridSearchCV
    """
    return [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10]
        },
        {
            "svm__kernel": ["rbg"],
            "svm__C": [0.1, 1, 10],
            "smv__gamma": ["scale", 0.01]
        }
    ]

def tune_model(pipeline, param_grid, X_train, y_train):
    """
    Tunes the pipeline via GridSearchCV.
    F1-Score -> OPtimization metric since dataset is imbalanced.
    Accuracy by itself would only reward the majority class prediction too heavily.
    --------------------------------------------------------
    INPUT:
        pipeline: (Pipeline) Full preprocessing + model pipeline
        param_grid: (list of dict) Search space for hyperparameters
        X_train: (pd.DataFrame) Training predictors
        y_train: (pd.Series) Training target

    OUTPUT:
        (GridSearchCV) Fitted grid search object
    """
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        # 5-fold cross-validation
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    return grid

def model_eval(model, X_test, y_test):
    """
    Evaluates best model on hold-out set.

    Reports on:
        Accuracy, Precision, REcall, F1-Score, and ROC-AUC
    ---------------------------------------------------------------
    INPUT:
        model: (Pipeline) Best fitted pipeline
        X_test: (pd.DataFrame) Test data
        y_test: (pd.Series) Target data

    OUTPUT:
        y_pred: (np.ndarray) Predicted class labels
        y_prob: (np.ndarray) Predicted positive-class probablities
    """
    # Explain these ??
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(dedent(f"""
+++++++++++++++++++++++++++++
MODEL PERFORMANCE ON TEST SET
+++++++++++++++++++++++++++++
ACCURACY: {accuracy_score(y_test, y_pred):.3f}
PRECISION: {precision_score(y_test, y_pred):.3f}
RECALL: {recall_score(y_test, y_pred):.3f}
F1-SCORE: {f1_score(y_test, y_pred):.3f}
ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}
                 """))

    # Output confusion matrix stuff
    print(dedent(f"""
CONFUSION MATRIX:
{confusion_matrix(y_test, y_pred)}

CLASSIFICATION REPORT:
{classification_report(y_test, y_pred)}
                 """))

    return y_test, y_pred

def plot_roc_curve(y_test, y_prob):
    """
    Plots ROC curve
    ----------------------------------------
    INPUT:
        y_test: (pd.Series) Test labels
        y_prob: (np.ndarray) +-class probabilities

    OUTPUT:
        None
    """
    # False-positive rates, True-positive rates
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="lime", linewidth=2, label=f"SVM ROC Curve (AUC={roc_auc}:.3f)")
    # ? Explain this shit ?
    plt.plot([0, 1], [0, 1], color="dodgerblue", linestyle="--", linewidth=1.73)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.grid(color="gray")
    plt.show()

def plot_confusion_heatmap(y_test, y_pred):
    """
    Plots confusion matrix and heatmap
    ------------------------------------------
    INPUT:
        y_test: (pd.Series) True labels
        y_pred: (np.ndarray) Predicted labels

    OUTPUT:
        None
    """
    # Confusion matrix calculation
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.grid(color="gray")
    plt.show()

def calc_permutation_importance(
    model,
    X_test,
    y_test,
    top_n=15
):
    """
    Computes the permutation importance on original input columns, where the
    function assumes the model is a sklearn Pipeline w/ both:
        - preprocessing
        - trained SVM.

    Randomization occurs in original feature space prior to preprocessing, meaning the resulting feature names correctly correspond to X_test.columns.
    ?
    --------------------------------------------------------
    INPUT:
        model: (Pipeline) Fitted preprocessing and SVM Pipeline
        X_test: (pd.DataFrame) Test data
        y_test: (pd.Series) True labels
        top_n: (int; default=15) Number of top features to plot

    OUTPUT:
        importance_df: (pd.DataFrame) Permutation importance summary table
    """
    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="f1",
        n_repeats=10,
        random_state=73,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importance_mean,
        "importance_std": result.importance_std
    }).sort_values(by="importance_mean", ascending=False)

    # top N features
    top_feats = importance_df.head(top_n).iloc[::-1]

    # Plotting
    plt.figure(figsize=(10, 7))
    # ??
    plt.barh(top_feats["feature"], top_features["importance_mean"])

    plt.xlabel("Mean Permutation IMportance (F1 Decreases)")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Permutation Importances")
    plt.tight_layout()
    plt.grid(color="gray")
    plt.show()

    return importance_df

def main():
    # Running the entire workflow

    # load data
    df = read_file("online_shoppers_intention.csv")

    # Split data
    X, y = split_feats_target(df, df.columns[-1])

    # Feature types
    numeric, categorical = get_feature_types(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=73
    )

    print(dedent(f"""
TRAIN/TEST SPLIT:
X_train shape: {X_train.shape}
X_test shape: {X_test.shape}
y_train shape: {y_train.shape}
y_test shape: {y_test.shape}
                 """))

    # Preprocessing, Pipeline, and parameter grid
    preproc = the_preprocessor(numeric, categorical)
    pipeline = build_pipeline(preproc)
    param_grid = define_param_grid()

    print("HYPERPARAMETER TUNING:")
    grid = tune_model(pipeline, param_grid, X_train, y_train)

    print(dedent(f"""\n
BEST PARAMETERS:
{grid.best_params_}

Best CV F1-Score: {grid.best_score_:.3f}
                 """))

    # ?? Kernel stuff ??

    plot_roc_curve(y_test, y_prob)

    plot_confusion_heatmap(y_test, y_pred)

    importance_df = calc_permuation_importance(
        best_model,
        X_test,
        y_test,
        top_n=15
    )

    print(dedent(f"""\n
TOP 15 PERMUTATION IMPORTANCES
{importance_df.head(15).to_string(index=False)}

FINAL INTERPRETATION
--------------------
¯\_( ͡° ͜ʖ ͡°)_/¯
                 """))


if __name__ == "__main__":
    main()
