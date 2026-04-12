# MLOps End-to-End Project

## Description

This project demonstrates how to build a complete MLOps system step by step, including data engineering, model training, deployment, and monitoring.

## Project Structure

```
mlops_-course/
├── data/
│   ├── raw/          # Raw ingested data
│   ├── processed/    # Cleaned and transformed data
│   └── incoming/     # New data arriving for inference or retraining
├── src/
│   ├── data/         # Data loading, validation, and preprocessing scripts
│   ├── pipelines/    # Pipeline orchestration scripts
│   ├── models/       # Model training and evaluation
│   ├── api/          # FastAPI model serving application
│   ├── monitoring/   # Model monitoring and drift detection
│   └── utils/        # Shared utility functions
├── notebooks/        # Experimentation and development notebooks
├── .github/
│   └── workflows/    # CI/CD pipeline definitions
└── requirements.txt  # Project dependencies
```

## Workflow Overview

```
Data Engineering → Training → Model Registry → Serving → Monitoring → Retraining
```

1. **Data Engineering** – ingest, validate, and preprocess raw data
2. **Training** – train and evaluate models, track experiments with MLflow
3. **Model Registry** – register and version the best model in MLflow
4. **Serving** – expose the model via a FastAPI REST endpoint
5. **Monitoring** – track prediction drift and model performance over time
6. **Retraining** – trigger automated retraining when performance degrades

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning |
| MLflow | Experiment tracking & model registry |
| FastAPI | Model serving |
| Docker *(optional)* | Containerization |

## Notes

This repository is part of an educational course. Code will be implemented progressively, step by step.
