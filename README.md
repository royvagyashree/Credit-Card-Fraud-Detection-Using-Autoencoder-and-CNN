#  Credit Card Fraud Detection Using Autoencoder and CNN

This project implements a **hybrid deep learning approach** using both an **Autoencoder** and a **Convolutional Neural Network (CNN)** to detect fraudulent transactions in credit card data. The model leverages reconstruction error from the Autoencoder and classification probability from the CNN for robust ensemble-based fraud detection.

---

##  Techniques Used

- **Autoencoder**: Learns the structure of legitimate transactions and flags fraud by high reconstruction error.
- **CNN**: Learns patterns from reshaped transactional features for direct classification.
- **SMOTE**: Handles class imbalance by oversampling minority class (fraud).
- **Ensemble Method**: Final probability is a weighted average of both models' outputs.

---

##  Contents

- `Jupyter Source Code.ipynb`: Main notebook including:
  - Data preprocessing
  - Feature scaling
  - SMOTE balancing
  - Autoencoder training
  - CNN training
  - Evaluation
  - Fraud prediction logic
  - Model saving

---

##  Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow keras joblib matplotlib seaborn
