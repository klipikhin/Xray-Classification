# COVID-19 Detection from Chest X-rays Using Deep Learning

This project applies Convolutional Neural Networks (CNNs) and transfer learning techniques to classify chest X-ray images into four categories:

- COVID-19
- Lung Opacity (non-COVID lung infection)
- Viral Pneumonia
- Normal

The goal is to assist medical professionals with fast and accurate diagnosis using AI, particularly in environments with limited radiological expertise.



##  Dataset

The dataset used is the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), developed by researchers from Qatar University, the University of Dhaka, and collaborating institutions.

**Class distribution:**

- COVID-19: 3,616 images  
- Lung Opacity: 6,012 images  
- Viral Pneumonia: 1,345 images  
- Normal: 10,102 images  

All images are in `.png` format and resized to 224x224 pixels during preprocessing.



##  Model Overview

I employed the **DenseNet121** architecture pre-trained on ImageNet for feature extraction. The final classifier consists of:

- Batch Normalization
- Dense layer (256 units, ReLU activation, L2 regularization)
- Dropout layer (rate = 0.3)
- Softmax output layer with 4 units

Other architectures evaluated include:
- **MobileNetV2**
- **InceptionV3**



##  Training Configuration

- **Optimizer:** Adamax  
- **Learning rate:** 0.0001  
- **Loss Function:** Categorical Cross-Entropy (with class weights to handle class imbalance)  
- **Metrics:** Accuracy, Recall  
- **EarlyStopping:** Applied with a patience of 3 epochs  

Data augmentation was applied to improve generalization:
- Random rotation
- Width/height shifts
- Shear and zoom transformations
- Horizontal flipping



##  Results

After training and evaluation, **DenseNet121** achieved the following performance on the validation set:

| Class             | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| COVID-19         | 0.93      | 0.89   | 0.91     |
| Lung Opacity     | 0.94      | 0.89   | 0.91     |
| Normal           | 0.91      | 0.96   | 0.93     |
| Viral Pneumonia  | 0.96      | 0.85   | 0.90     |

**Final Validation Accuracy:** 92%
**COVID-19 Recall:** 89%

The model was most challenged by distinguishing COVID-19 from Lung Opacity due to similar visual features.



##  Project Report

You can read the full technical report here:  
📘 [View Report](<Report.pdf>)



##  Presentation Video

A short presentation explaining the motivation, methodology, and results of this project is available on YouTube:  
▶️ [Watch Presentation](<https://youtu.be/mj8EsPgKqSI>)

---

