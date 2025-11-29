## Project Overview

This repository contains a small end-to-end **MLOps pipeline** for the Boston Housing dataset, built for the *Cloud MLOps* course assignment.

The pipeline:
- Uses **DVC** for data versioning of the raw Boston Housing CSV.
- Builds reusable **Kubeflow Pipelines components** for:
  - data extraction
  - data preprocessing (scaling + train/test split + binary target)
  - model training (Random Forest classifier)
  - model evaluation (accuracy & F1-score)
- Orchestrates these components as a full Kubeflow pipeline on **Minikube**.
- Uses a **Jenkins declarative pipeline** to automatically set up the environment and compile `pipeline.yaml`.

The ML task is to predict whether a house price (`medv`) is **above or below the median** price, i.e. a **binary classification** problem derived from the classic regression dataset.

---

## Setup Instructions

### 1. Local Python Environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `pandas`, `numpy`, `scikit-learn`, `joblib`
- `kfp` (Kubeflow Pipelines SDK)
- `dvc`

### 2. DVC and Data Remote

This repo is already initialized with DVC and contains `data/raw_data.csv` plus its corresponding `.dvc` tracking file.

To re-create or inspect the setup:

```bash
cd mlops-kubeflow-assignment
source .venv/bin/activate

# Initialize DVC (already done in this repo)
dvc init

# Local remote location (already configured)
mkdir -p dvc_remote
dvc remote add -d localremote dvc_remote

# Add data and push to remote
dvc add data/raw_data.csv
dvc status
dvc push
```

`data/raw_data.csv` is the full Boston Housing dataset (506 rows) fetched from scikit-learn and stored as a CSV.

### 3. Minikube and Kubeflow Pipelines

1. **Start Minikube** (example configuration; adjust resources as needed):

```bash
minikube start --cpus=4 --memory=8192
minikube status
```

2. **Install Kubeflow Pipelines** (standalone or as directed in the course).
   - Follow the official KFP / course instructions.
   - Ensure the KFP UI and backend pods are running in the `kubeflow` namespace:

```bash
kubectl get pods -n kubeflow
```

3. **Access the KFP UI**:
   - Typically via a port-forward or Minikube service tunnel (depending on the installation).
   - Open the URL in your browser (often `http://localhost:8080` or another forwarded port).

---

## Pipeline Walkthrough

### 1. Generate Components and Pipeline Package

From the repository root:

```bash
source .venv/bin/activate

# Generate/refresh Kubeflow component YAMLs under components/
python src/pipeline_components.py

# Compile the main KFP pipeline to pipeline.yaml
python pipeline.py
```

This produces:
- `components/data_extraction_component.yaml`
- `components/data_preprocessing_component.yaml`
- `components/model_training_component.yaml`
- `components/model_evaluation_component.yaml`
- `pipeline.yaml` in the project root

### 2. Upload and Run the Pipeline in KFP

1. Open the **Kubeflow Pipelines UI**.
2. Go to **Pipelines → Upload** and upload `pipeline.yaml` (create a new pipeline).
3. Create a **new run** of the uploaded pipeline:
   - Use the default parameters, but ensure `repo_url` points to your **GitHub repository URL** for this project.
4. Wait for all four steps to complete:
   - `data-extraction`
   - `data-preprocessing`
   - `model-training`
   - `model-evaluation`

The `model-evaluation` step prints metrics (accuracy and F1-score) in its logs and writes them to `metrics/metrics.json` inside the container.

---

## Continuous Integration (Jenkins)

The `Jenkinsfile` in the project root defines a **declarative Jenkins pipeline** with the following stages:

1. **Checkout** – pulls the source code from SCM (GitHub).
2. **Environment Setup** – creates a Python virtual environment, upgrades `pip`, and installs dependencies from `requirements.txt`.
3. **Pipeline Compilation** – runs:
   - `python src/pipeline_components.py` to generate component YAML files.
   - `python pipeline.py` to compile the full Kubeflow pipeline into `pipeline.yaml`.
4. **Sanity Check** – verifies that `pipeline.yaml` exists and lists the `components/` directory.

To use this in Jenkins:
- Create a **Pipeline** job.
- Set **Definition → Pipeline script from SCM**, select **Git**, and point the **Repository URL** to your GitHub repo.
- Keep the script path as `Jenkinsfile`.
- Build the job and verify that all stages complete successfully.

---

## Repository Structure (Summary)

- `data/`
  - `raw_data.csv` – full Boston Housing dataset (DVC-tracked).
- `src/`
  - `pipeline_components.py` – Python functions for each KFP component and YAML compilation utility.
  - `model_training.py` – standalone training script (optional).
- `components/` – generated Kubeflow component YAMLs.
- `pipeline.py` – main Kubeflow pipeline definition (compiled into `pipeline.yaml`).
- `pipeline.yaml` – compiled pipeline package for upload to KFP.
- `Jenkinsfile` – Jenkins declarative CI pipeline.
- `requirements.txt` – Python dependencies.
- `.dvc/`, `data/raw_data.csv.dvc`, `dvc_remote/` – DVC metadata and local data remote.

---

## How to Reproduce the Full Flow

1. Clone the repository and create the Python environment.
2. Use DVC to ensure `data/raw_data.csv` is present (or re-generate via the provided script logic).
3. Start Minikube and deploy Kubeflow Pipelines.
4. Run:
   - `python src/pipeline_components.py`
   - `python pipeline.py`
5. Upload `pipeline.yaml` to KFP and create a run.
6. (Optional) Configure the Jenkins job to automatically perform steps 4–5 on each commit.


