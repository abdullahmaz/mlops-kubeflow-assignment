"""
Main Kubeflow pipeline definition for the Boston housing ML project.

This pipeline orchestrates the four components defined in
`src/pipeline_components.py` and compiled to YAML under `components/`:

- Data extraction (DVC-backed dataset fetch).
- Data preprocessing (scaling + train/test split).
- Model training (Random Forest classifier).
- Model evaluation (accuracy + F1-score).

Running this file as a script will compile the pipeline into `pipeline.yaml`,
which you can upload to the Kubeflow Pipelines UI.
"""

from kfp import dsl, components as kfp_components


# Load component definitions from the generated YAML files.
data_extraction_comp = kfp_components.load_component_from_file(
    "components/data_extraction_component.yaml"
)
data_preprocessing_comp = kfp_components.load_component_from_file(
    "components/data_preprocessing_component.yaml"
)
model_training_comp = kfp_components.load_component_from_file(
    "components/model_training_component.yaml"
)
model_evaluation_comp = kfp_components.load_component_from_file(
    "components/model_evaluation_component.yaml"
)


@dsl.pipeline(
    name="boston-housing-ml-pipeline",
    description="End-to-end pipeline: DVC data extraction, preprocessing, training, evaluation.",
)
def boston_housing_pipeline(
    repo_url: str = "https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git",
    dvc_data_path: str = "data/raw_data.csv",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = -1,
    random_state: int = 42,
):
    """
    Define the end-to-end Kubeflow pipeline for the Boston housing project.

    Parameters
    ----------
    repo_url : str
        Git repository URL that contains the DVC-tracked dataset.
        NOTE: Replace `YOUR_USERNAME` with your actual GitHub username
        before running this on Kubeflow.
    dvc_data_path : str
        Path to the DVC-tracked CSV file within the repository.
    test_size : float
        Fraction of data used for the test split.
    n_estimators : int
        Number of trees used by the Random Forest classifier.
    max_depth : int
        Maximum depth of each tree (-1 to indicate \"no limit\").
    random_state : int
        Random seed used for reproducible splitting and training.
    """

    # 1) Data extraction: fetch the versioned dataset using DVC.
    extract_task = data_extraction_comp(
        repo_url=repo_url,
        dvc_data_path=dvc_data_path,
        output_csv_path="data/raw_data.csv",
    )

    # 2) Data preprocessing: scaling + train/test split.
    preprocess_task = data_preprocessing_comp(
        raw_data_csv="data/raw_data.csv",
        x_train_path="data/processed_X_train.csv",
        x_test_path="data/processed_X_test.csv",
        y_train_path="data/processed_y_train.csv",
        y_test_path="data/processed_y_test.csv",
        test_size=test_size,
        random_state=random_state,
    )
    # Ensure preprocessing runs after data extraction.
    preprocess_task.after(extract_task)

    # 3) Model training: Random Forest classifier.
    # NOTE: KFP v2 does not allow passing `None` as a constant input value.
    # To avoid that, we always pass a positive integer for `max_depth`.
    # You can tune this value as needed when running the pipeline.
    train_task = model_training_comp(
        x_train_csv="data/processed_X_train.csv",
        y_train_csv="data/processed_y_train.csv",
        model_output_path="model/random_forest_model.joblib",
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    # Ensure training runs after preprocessing.
    train_task.after(preprocess_task)

    # 4) Model evaluation: accuracy + F1-score on the test set.
    eval_task = model_evaluation_comp(
        model_path="model/random_forest_model.joblib",
        x_test_csv="data/processed_X_test.csv",
        y_test_csv="data/processed_y_test.csv",
        metrics_output_path="metrics/metrics.json",
    )
    # Ensure evaluation runs after training.
    eval_task.after(train_task)


if __name__ == "__main__":
    # Compile the pipeline to `pipeline.yaml`.
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path="pipeline.yaml",
    )

