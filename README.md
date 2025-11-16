# Git, GitHub & GitHub Actions Demo

A comprehensive demo repository showing Git basics, GitHub collaboration, and GitHub Actions CI/CD with a machine learning project.

## Project Overview

This project implements a simple Logistic Regression classifier for the Iris dataset from scikit-learn.

## Features

- Data loading and preprocessing
- Logistic Regression model training
- Model evaluation and persistence
- Automated testing with GitHub Actions
- CI/CD pipeline for model training

## Repository Structure
```
git-github-actions-demo/
├── .github/workflows/ # GitHub Actions workflows
├── src/ # Source code
├── tests/ # Test cases
├── models/ # Trained models
├── requirements.txt # Dependencies
├── train.py # Training script
└── predict.py # Prediction script
```

## Git Basics

### Common Git Commands

```
# Clone repository
git clone <repository-url>

# Check status
git status

# Add files to staging
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push to remote
git push origin main

# Create and switch to new branch
git checkout -b feature/new-feature

# Merge branches
git merge feature/new-feature
```

## GitHub Actions
This repository includes two workflows:

- CI/CD Pipeline - Runs on push/pull requests to main branch
- Model Testing - Runs model training and testing

# Getting Started

1. Install dependencies
```
$pip install -r requirements.txt
```

2. Train the model
```
$python src/train.py
```
3. Make predictions
```
$python src/predict.py
```
