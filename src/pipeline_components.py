"""
Kubeflow pipeline component definitions for the Boston housing ML project.

This module defines the core building blocks of the ML pipeline as reusable
Kubeflow components:

- Data extraction (fetch versioned dataset using DVC).
- Data preprocessing (cleaning, scaling, train/test split).
- Model training (Random Forest classifier).
- Model evaluation (compute accuracy and F1-score).

Running this file as a script will also compile these functions into
stand-alone Kubeflow Component YAML files under the `components/` directory.
"""

import os
import json
import subprocess
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

from kfp import dsl


def data_extraction(
    repo_url: str,
    dvc_data_path: str,
    output_csv_path: str = "data/raw_data.csv",
) -> None:
    """
    Fetch the versioned dataset from a DVC-enabled repository.

    Parameters
    ----------
    repo_url : str
        URL of the Git repository that contains the DVC-tracked data
        (e.g., your GitHub repository URL).
    dvc_data_path : str
        Path (within the repo) to the data file or directory to fetch,
        for example: \"data/raw_data.csv\".
    output_csv_path : str, optional
        Local path inside the component container where the CSV file
        should be stored.

    Notes
    -----
    The dataset is written to ``output_csv_path`` inside the component container.
    """
    import subprocess
    # Use `dvc get` to download the data file from the remote repo.
    # Example:
    #   dvc get https://github.com/user/mlops-kubeflow-assignment data/raw_data.csv -o /tmp/raw_data.csv
    cmd = [
        "dvc",
        "get",
        repo_url,
        dvc_data_path,
        "-o",
        output_csv_path,
    ]
    subprocess.run(cmd, check=True)


def data_preprocessing(
    raw_data_csv: str,
    x_train_path: str = "data/processed_X_train.csv",
    x_test_path: str = "data/processed_X_test.csv",
    y_train_path: str = "data/processed_y_train.csv",
    y_test_path: str = "data/processed_y_test.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Load the raw dataset, clean, scale, and split it into train/test sets.

    The Boston housing dataset is a regression dataset. For the purposes of
    this assignment (which asks for a classifier and metrics like accuracy
    and F1-score), we convert the target into a binary label:

    - target = 1 if MEDV >= median(MEDV)
    - target = 0 otherwise

    Parameters
    ----------
    raw_data_csv : str
        Path to the raw CSV file containing the full Boston housing dataset.
    x_train_path, x_test_path, y_train_path, y_test_path : str
        Paths where the processed train/test splits will be stored as CSV.
    test_size : float
        Fraction of the data to reserve for testing.
    random_state : int
        Random seed for reproducible splits.

    Notes
    -----
    The processed splits are written to the four output CSV paths.
    """
    df = pd.read_csv(raw_data_csv)

    if "medv" not in df.columns:
        raise ValueError("Expected target column 'medv' in the dataset.")

    # Create binary classification target from regression target.
    median_medv = df["medv"].median()
    df["target"] = (df["medv"] >= median_medv).astype(int)

    feature_cols = [c for c in df.columns if c not in ("medv", "target")]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features using StandardScaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed splits to CSV.
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(x_train_path, index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False, header=True)
    y_test.to_csv(y_test_path, index=False, header=True)


def model_training(
    x_train_csv: str,
    y_train_csv: str,
    model_output_path: str = "model/random_forest_model.joblib",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> None:
    """
    Train a Random Forest classifier on the preprocessed training data.

    Parameters
    ----------
    x_train_csv : str
        Path to the preprocessed feature matrix for training.
    y_train_csv : str
        Path to the training labels.
    model_output_path : str
        Path where the trained model artifact will be stored.
    n_estimators : int
        Number of trees in the forest.
    max_depth : Optional[int]
        Maximum depth of each tree.
    random_state : int
        Random seed for reproducibility.

    Notes
    -----
    The trained model is saved to ``model_output_path``.
    """
    X_train = pd.read_csv(x_train_csv)
    y_train = pd.read_csv(y_train_csv).iloc[:, 0]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)


def model_evaluation(
    model_path: str,
    x_test_csv: str,
    y_test_csv: str,
    metrics_output_path: str = "metrics/metrics.json",
) -> None:
    """
    Evaluate the trained model on the test set and save metrics to a file.

    Metrics include:
    - accuracy
    - F1-score (binary, positive class = 1)

    Parameters
    ----------
    model_path : str
        Path to the trained model artifact.
    x_test_csv : str
        Path to the test feature matrix.
    y_test_csv : str
        Path to the test labels.
    metrics_output_path : str
        Path where the JSON metrics file will be written.

    Notes
    -----
    The JSON metrics file is written to ``metrics_output_path``.
    """
    X_test = pd.read_csv(x_test_csv)
    y_test = pd.read_csv(y_test_csv).iloc[:, 0]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {"accuracy": acc, "f1_score": f1}

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Also print metrics so they are visible in KFP logs.
    print(json.dumps(metrics, indent=2))


def compile_kubeflow_components() -> None:
    """
    Compile the Python functions into Kubeflow component YAML files.

    The generated YAML files are written into the `components/` directory:
    - data_extraction_component.yaml
    - data_preprocessing_component.yaml
    - model_training_component.yaml
    - model_evaluation_component.yaml
    """
    os.makedirs("components", exist_ok=True)

    # Note: base_image and packages_to_install ensure that each component
    # container has the required runtime dependencies.
    common_kwargs = {
        "base_image": "python:3.11-slim",
        "packages_to_install": [
            "pandas",
            "numpy",
            "scikit-learn",
            "joblib",
            "dvc",
        ],
    }

    factory = dsl.component_factory

    data_extraction_comp = factory.create_component_from_func(
        data_extraction,
        **common_kwargs,
    )
    data_extraction_comp.component_spec.save_to_component_yaml(
        "components/data_extraction_component.yaml"
    )

    data_preprocessing_comp = factory.create_component_from_func(
        data_preprocessing,
        **common_kwargs,
    )
    data_preprocessing_comp.component_spec.save_to_component_yaml(
        "components/data_preprocessing_component.yaml"
    )

    model_training_comp = factory.create_component_from_func(
        model_training,
        **common_kwargs,
    )
    model_training_comp.component_spec.save_to_component_yaml(
        "components/model_training_component.yaml"
    )

    model_evaluation_comp = factory.create_component_from_func(
        model_evaluation,
        **common_kwargs,
    )
    model_evaluation_comp.component_spec.save_to_component_yaml(
        "components/model_evaluation_component.yaml"
    )


if __name__ == "__main__":
    # When run as a script, generate all component YAML files into `components/`.
    compile_kubeflow_components()

