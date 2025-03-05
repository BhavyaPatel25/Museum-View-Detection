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

```python
pip install numpy scikit-learn pillow
```

## Instruction to train/validate model

### Training the Model

To train the model, follow these steps:

1. **Load the dataset:** Ensure the dataset is properly preprocessed before training. For loading the dataset use [Museum Train](museum_train.zip) and before preprocessing extract it.
2. **Split the data:** Divide the dataset into training and validation sets, typically using an 80-20 split.

    ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Initialize the model:** Choose the appropriate machine learning model based on the task.
5. **Train the model:** Fit the model to the training data.

    ```python
   model.fit(X_train, y_train)
    ```

### Validating the Model

After training, evaluate the model's performance using appropriate metrics:

1. **Make predictions on the test set:**
   
     ```python
   y_pred = model.predict(X_test)
     ```
   
2. **Compute evaluation metrics:**
   
    ```python
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1score * 100:.2f}%')
    ```
3. **Display the confusion matrix:**

    ```python
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['indoor', 'outdoor'])
    ```
    This helps visualize the model’s classification performance by showing the distribution of true and predicted labels.
       

## Instruction to run our pretrained model on test dataset

1. Extract the pretrained models from [Models.zip](Models.zip)
2. Load the model using the below given code

    ```python
    import pickle
    
    def load_model_pickle(path="model.pkl"):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model

    model = load_model_pickle('path_to_model.pkl')
    ```
3. Use the preprocessing function given below

    ```python
    from PIL import Image
    import numpy as np
    
    def load_single_images(image):
        img = Image.open(image)
        img = img.resize((64, 64))
        img = img.convert('L')
        img = np.array(img).flatten()
        img = img/255
        return img

    test_image = load_single_images('path_to_test_image.jpg')
    ```
4. Use `model.predict(test_image)` for testing the model.
5. If `prediction == 0` then the image belongs to Indoor Museum Class, else if `prediction == 1` then image belongs to Outdoor Museum Class.

## How to obtain the dataset

Dataset Source: [http://places2.csail.mit.edu/download.html](http://places2.csail.mit.edu/download.html)

1. Visit the source link.
2. Click on `Sign this form`.
3. Fill the form.
4. You will be redirectly to Form Submission page where there are multiple links.
5. Open the first link and there you can find various version of the dataset that can be downloaded.

## Credits

**Dataset:** A 10 million Image Database for Scene Recognition 
B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017
