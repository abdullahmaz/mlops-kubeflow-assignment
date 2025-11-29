pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        // Name of the Python virtual environment directory used in the build workspace
        VENV_DIR = ".venv"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code from SCM...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Setting up Python environment and installing dependencies...'
                sh '''
                    set -e

                    # Use system python3; Jenkins agent should have it installed.
                    python3 --version

                    # Create / reuse virtual environment
                    if [ ! -d "$VENV_DIR" ]; then
                      python3 -m venv "$VENV_DIR"
                    fi

                    . "$VENV_DIR/bin/activate"
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline to pipeline.yaml...'
                sh '''
                    set -e
                    . "$VENV_DIR/bin/activate"

                    # Generate or refresh the component YAML files
                    python src/pipeline_components.py

                    # Compile the main Kubeflow pipeline definition
                    python pipeline.py
                '''
            }
        }

        stage('Sanity Check') {
            steps {
                echo 'Running a quick sanity check on the compiled artifacts...'
                sh '''
                    set -e
                    ls -lah
                    test -f pipeline.yaml
                    ls -lah components
                '''
            }
        }
    }

    post {
        success {
            echo 'Jenkins pipeline completed successfully. Kubeflow pipeline.yaml is ready.'
        }
        failure {
            echo 'Jenkins pipeline failed. Check the stage logs above for details.'
        }
    }
}

