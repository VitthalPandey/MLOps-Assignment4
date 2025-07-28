# MLOps Linear Regression Pipeline

This project implements an end-to-end MLOps pipeline using a simple Linear Regression model to predict housing prices from the California Housing dataset. It includes model training, evaluation, parameter quantization, testing, containerization, and automation through continuous integration.

# Project Objective

To showcase the integration of machine learning operations (MLOps) principles by:

Building a reproducible training pipeline

Applying manual quantization techniques

Automating testing and build processes

Containerizing the workflow for portability

Leveraging GitHub Actions for CI/CD automation

# Directory Overview

Folder/File	Description
src/	Contains all source code and utilities
tests/	Holds test cases written using pytest
.github/workflows/	CI workflow configuration
Dockerfile	Docker image definition
requirements.txt	Python dependency list
README.md	Project overview and instructions

# Continuous Integration
This project uses GitHub Actions for automated CI. On every push to the main branch, the following tasks are triggered:

Model training and saving

Unit test execution

Quantization script execution

Docker container build and run check

The configuration is defined in:
.github/workflows/ci.yml

# Docker Instructions
Build and run the containerized version:

bash
Copy
Edit
docker build -t mlops-lr .
docker run mlops-lr

# Evaluation Metrics
Metric	Description
R² Score	Coefficient of determination
Mean Squared Error	Average squared prediction error

# Technologies Used
Python 3.10+

scikit-learn

joblib

pytest

Git & GitHub

Docker

GitHub Actions

# Information 

Name: Vitthal Pandey
Roll Number: G24AI1097
Program: PGD DE
Institute: IIT Jodhpur







