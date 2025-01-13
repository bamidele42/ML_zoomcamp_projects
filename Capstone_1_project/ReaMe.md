# Customer Default Payment Prediction Project

## Overview

This project aims to analyze the default payment behavior of customers in Taiwan and build a predictive model to estimate the probability of default among customers. The model leverages machine learning techniques and provides a deployed solution for real-time predictions.

---

## Project Workflow

### Jupyter Notebook Workflow

The machine learning pipeline was developed using Jupyter Notebook. Key steps in the workflow include:

- **Data Cleaning**: Ensuring data quality by handling missing values, outliers, and inconsistencies.
- **Exploratory Data Analysis (EDA)**: Visualizing and summarizing data to uncover patterns and insights.
- **Feature Engineering**: Creating and selecting relevant features for better model performance.
- **Model Training**: Building and evaluating machine learning models to predict customer defaults.

---

## Prediction App

The prediction application was built using **FastAPI**, a modern and high-performance web framework. The application:

- Allows for seamless and efficient deployment of the predictive model.
- Utilizes **Pydantic** BaseModel to integrate data validation and type checking into the API.

---

## Deployment

### FastAPI

**FastAPI** was chosen as the development environment due to its speed and efficiency. It is based on standard Python type hints, making it a robust framework for building APIs.

---

### Containerization

The application was containerized using Docker for consistent and scalable deployment.

#### Steps to Build and Run the Docker Container:

1. **Build the Docker image**:
   ```bash
   docker build -t default_payment .
   ```
2. run the docker container
   docker run --name container_name -p 8000:8000 default_payment
