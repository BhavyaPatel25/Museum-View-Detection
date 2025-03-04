# Museum Detection

## Description

The project focuses on solving an image classification problem using machine learning and deep learning techniques. The primary goal is to classify images as either indoor or outdoor using Decision Trees, Random Forests, Boosting, and Convolutional Neural Networks (CNNs). The project involves both supervised and semi-supervised learning approaches.

The dataset consists of images from the MIT Places dataset, specifically museum indoor and outdoor images. 
http://places.csail.mit.edu/browser.html

The project is divided into two phases:

### Phase 1: Decision Trees & Boosting Models
* Implement Decision Tree, Random Forest, and Boosting (Gradient Boosting or XGBoost) models using supervised learning.
* Implement a semi-supervised Decision Tree model, where unlabeled data is progressively labeled based on model confidence scores.
* Perform exploratory data analysis (EDA) and image preprocessing to optimize the dataset for training.
* Optimize the models using hyperparameter tuning (e.g., tree depth, number of branches, pruning).
* Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix.

### Phase 2: Convolutional Neural Network (CNN) Model
* Design and implement a CNN model from scratch using PyTorch.
* Train and validate the CNN model using the dataset, ensuring an automated train-test split.
* Optimize CNN hyperparameters (e.g., number of convolutional layers, pooling layers).
* Compare the CNN’s performance against Decision Trees, Random Forests, and Boosting models.
* The trained CNN should be able to classify individual images in real-time.

## Requirements to run the code

**Python Version:** Ensure you have Python 3.8+ installed.

**Jupyter Notebook:** You need Jupyter Notebook or JupyterLab to run the .ipynb files.

### Required Python Libraries

You need to install the following Python packages before running the notebooks:

`pip install numpy scikit-learn pillow`

## Instruction to train/validate model


## Instruction to run our pretrained model on test dataset

## How to obtain the dataset
