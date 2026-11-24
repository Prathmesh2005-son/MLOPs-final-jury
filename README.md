# P10: CI/CD Pipeline for an ML Project

Automated MLOps pipeline using **GitHub Actions** that:
1. Lints Python code using lake8
2. Runs unit tests using pytest
3. Trains a RandomForest classifier on the Iris dataset
4. Enforces an accuracy **Quality Gate** (>= 0.85)
5. Builds and pushes a Docker image to **GitHub Container Registry (ghcr.io)**
