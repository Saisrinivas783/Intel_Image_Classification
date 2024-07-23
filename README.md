# Intel Image Classification

This project explores the process of building and evaluating image classification models using Convolutional Neural Networks (CNNs) and the Keras library. The main goal is to classify images into six categories: mountain, street, glacier, buildings, sea, and forest.

## Project Overview

1. **Project Setup**: 
   - Imported necessary libraries and packages for image processing, neural network construction, and data manipulation.

2. **Data Preparation**: 
   - Loaded the image dataset, visualized a sample, and resized and normalized the pixel values.

3. **Simple CNN Implementation**: 
   - Designed a basic CNN architecture with convolutional, max pooling, and dense layers.
   - Trained this model on the prepared image data.

4. **Model Evaluation**: 
   - Evaluated the initial CNN model using performance metrics like accuracy.
   - Conducted basic error analysis to identify potential areas for improvement.

5. **Leveraging Transfer Learning**: 
   - Enhanced model performance by fine-tuning a pre-trained VGG16 model.
   - Experimented with freezing and training different layers of the pre-trained model.

## Data Loading

The `load_data()` function loads images and their labels from the predefined training and testing directories:

- **Training Dataset**: 14,034 images
- **Testing Dataset**: 3,000 images

## Exploratory Data Analysis

- Analyzed the distribution of images across different classes using bar and pie charts.
- Scaled the image pixel values to a range of 0-1 for better training.

## Simple CNN Model

- **Architecture**:
  - Two convolutional layers with 32 filters each, followed by max pooling layers.
  - A flattening layer and two dense layers for classification.

- **Compilation**:
  - Optimizer: Adam
  - Loss Function: Sparse Categorical Crossentropy

- **Training**:
  - Trained the model with a batch size of 128 for 10 epochs.

## Model Evaluation

- Achieved an accuracy of 0.76 on the test dataset, with slight underfitting.
- Performed error analysis to understand misclassified images using confusion matrices.

## Transfer Learning with VGG16

- Extracted features using VGG16 and performed PCA to visualize feature clusters.
- Trained a simple one-layer neural network on VGG-extracted features, improving accuracy.

## Ensemble Learning

- Implemented an ensemble of 10 neural networks trained on random subsets of the data.
- Aggregated predictions from all models to improve classification performance.

## Fine-Tuning VGG16

- Fine-tuned VGG16 by unfreezing the last few layers and adding additional layers for classification.
- Achieved improved accuracy through fine-tuning.

## Key Findings

- **Challenges**:
  - Difficulty distinguishing between street and building images, as well as between sea, glacier, and mountain images.
  
- **Successes**:
  - High accuracy in identifying forest images.

## Future Work

- Experiment with other pre-trained models like ResNet or Inception for potentially better performance.
- Further investigate data augmentation techniques to enhance model robustness.
