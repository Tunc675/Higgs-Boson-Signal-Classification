# Higgs Boson Signal Classification

This project is about classifying Higgs boson signals versus background events using deep learning.  
It was created as a learning and practice project, but it is also useful to understand how real physics data can be connected with machine learning.

---

## Datasets

I used **two different datasets** for this project:

1. **ATLAS Higgs Boson dataset (from Kaggle / CERN challenge)**  
   - This dataset is very clean and well-prepared.  
   - The model reached very high accuracy (around 99%) because there is not much noise.  

2. **Higgs.csv dataset (from HuggingFace)**  
   - This dataset is more noisy and realistic.  
   - The model accuracy was lower (around 62%) but this is normal because the data is closer to real physics experiments.  

---

## Project Pipeline

1. **Data Preprocessing**
   - Replace missing values (`?`) with 0.  
   - Normalize features between 0 and 1.  
   - Train/Validation/Test split (80/10/10).  

2. **Neural Network Model**
   - Multi-layer dense neural network (MLP).  
   - Layers: Dense + Batch Normalization + Dropout + L2 Regularization.  
   - Optimizer: Adam with low learning rate.  
   - Loss: Binary Focal Crossentropy.  

3. **Training**
   - EarlyStopping and ModelCheckpoint to avoid overfitting.  
   - Class weights are used to handle data imbalance.  

4. **Evaluation**
   - Confusion Matrix  
   - Precision–Recall Curve (with best threshold point)  
   - ROC Curve (AUC score around 0.66–0.67 on noisy dataset)  

---

## Results

- **ATLAS dataset** → Accuracy ≈ 0.99 (because dataset is clean).  
- **Higgs.csv dataset** → Accuracy ≈ 0.62 (because dataset has more noise).  

**Confusion Matrix Example:**  
Correct classification of both signal and background events, but with some misclassifications due to noise.  

**Precision–Recall Curve:**  
Shows the trade-off between capturing true signals (recall) and avoiding false detections (precision).  

**ROC Curve:**  
Shows how well the model separates signal vs background. AUC ≈ 0.67 means the model is better than random guessing but still not perfect.

---

## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/Tunc675/Higgs-Boson-Signal-Classification.git
   cd Higgs-Boson-Signal-Classification
   
## Future Work

- Extend the current binary classification into a **multi-class classification** problem  
  (e.g., distinguish between different Higgs decay channels and background processes).

- Experiment with different neural network architectures such as **Convolutional Neural Networks (CNNs)**  
  and **Recurrent Neural Networks (RNNs)** to test their performance on structured physics data.

- Perform **hyperparameter tuning** (learning rate, batch size, dropout rates, etc.)  
  to improve performance on noisy datasets.

- Compare results with other machine learning methods (e.g., Gradient Boosting, Random Forests)  
  for a stronger baseline.

## Notes

- This project was developed mainly for **learning and practice purposes**.  
- It connects concepts from **high-energy physics (Higgs boson)** with **machine learning (deep learning)**.  
- Results show that:
  - On clean datasets (ATLAS), the model can reach very high accuracy (~99%).  
  - On noisy datasets (Higgs.csv), accuracy is lower (~62%), but the evaluation curves (Precision–Recall, ROC) are more meaningful.  
- The project demonstrates the challenge of **working with noisy, real-like data** in particle physics.  
- Accuracy alone is not enough: metrics such as **F1-score, AUC, Precision–Recall curves** give deeper insights.  
