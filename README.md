**# MLOps Linear Regression Pipeline**

This project implements an end-to-end MLOps pipeline using a simple Linear Regression model to predict housing prices from the California Housing dataset. It includes model training, evaluation, parameter quantization, testing, containerization, and automation through continuous integration.

**# Project Objective**

To showcase the integration of machine learning operations (MLOps) principles by:

1. Building a reproducible training pipeline

2. Applying manual quantization techniques

3. Automating testing and build processes

4. Containerizing the workflow for portability

5. Leveraging GitHub Actions for CI/CD automation

**# Directory Overview **

| Folder/File        | Description                            |
| ------------------ | -------------------------------------- |
| src/               | Contains all source code and utilities |
| tests/             | Holds test cases written using pytest  |
| .github/workflows/ | CI workflow configuration              |
| Dockerfile         | Docker image definition                |
| requirements.txt   | Python dependency list                 |
| README.md          | Project overview and instructions      |


**# Continuous Integration**

This project uses GitHub Actions for automated CI. On every push to the main branch, the following tasks are triggered:

1. Model training and saving

2. Unit test execution

3. Quantization script execution

4. Docker container build and run check

The configuration is defined in:
.github/workflows/ci.yml

**# Docker Instructions**

Build and run the containerized version:

docker build -t mlops-lr .
docker run mlops-lr

**# Evaluation Metrics**

| Metric             | Description                      |
| ------------------ | -------------------------------- |
| R² Score           | Coefficient of determination     |
| Mean Squared Error | Average squared prediction error |

**# Comparison Table**

| Step             | Description                        | Method/Tool Used           | Output Artifact          |
| ---------------- | ---------------------------------- | -------------------------- | ------------------------ |
| Data Loading     | Load California Housing dataset    | scikit-learn               | X (features), y (target) |
| Model Training   | Fit Linear Regression model        | LinearRegression (sklearn) | model.joblib             |
| Model Evaluation | Assess performance                 | R² Score, MSE              | Printed metrics          |
| Model Testing    | Validate trained model             | pytest                     | test logs                |
| Parameter Saving | Save model weights                 | joblib                     | model.joblib             |
| Quantization     | Manually reduce model precision    | NumPy operations           | quant\_params.joblib     |
| Inference        | Predict using quantized parameters | Custom predict.py          | Console predictions      |
| Dockerization    | Package workflow as container      | Dockerfile                 | Docker image             |
| CI/CD Automation | Automate testing & build on push   | GitHub Actions             | Workflow execution logs  |


**# Technologies Used**

1. Python 3.10+

2. scikit-learn

3. joblib

4. pytest

5. Git & GitHub

6. Docker

7. GitHub Actions

**# Information **

Name: Vitthal Pandey
Roll Number: G24AI1097
Program: PGD DE
Institute: IIT Jodhpur







