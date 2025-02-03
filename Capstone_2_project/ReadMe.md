# Dog and Cat Image Classification Using Convolutional Neural Networks (CNN)

## Introduction

### Problem Overview

This project focuses on the development and implementation of a Convolutional Neural Network (CNN) classifier designed to distinguish between dog and cat images. The dataset consists of over 1,000 images of cats and dogs, scraped from Google Images. The primary objective is to create a model capable of accurately classifying images as either containing a cat or a dog.

The images in the dataset vary in size, ranging from approximately 100x100 pixels to 2000x1000 pixels, and are in JPEG format. The dataset is sourced from [Kaggle - Cats and Dogs Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification).

### Workflow using Kaggle Notebook

The workflow for this project involved resizing images to (150, 150) pixels for training the model, and (299, 299) pixels for evaluating the model's performance. Key steps in the process include:

- Hyperparameter tuning for various learning rates, layer depths, and dropout rates to optimize the model’s performance.
- Data augmentation techniques were applied to enhance model accuracy.
- Once a satisfactory level of accuracy was achieved, the model was saved for further use.

Kaggle's GPU resources were utilized throughout the project to facilitate efficient model training, leveraging the powerful computing capabilities available on the platform.

### Deployment

The deployment phase involved the following steps:

1. **Model Conversion:**  
   The trained model was first converted into a lighter format using TensorFlow Lite. TensorFlow Lite offers optimized models for mobile and embedded devices, ensuring faster inference.

2. **AWS Lambda Deployment:**  
   A Python script was developed to deploy the saved TensorFlow Lite model onto AWS Lambda. AWS Lambda provides a serverless computing environment, allowing the model to be deployed and executed without the need for managing server infrastructure.

### Containerization

To ensure consistency and simplify the deployment process, a Dockerfile was created to containerize the application. Docker enables the encapsulation of all dependencies, ensuring the application runs smoothly in different environments. The Dockerfile includes the installation of all necessary dependencies for the model and its deployment.
