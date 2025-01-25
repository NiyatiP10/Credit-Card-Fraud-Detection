### Credit Card Fraud Detection

This project aims to detect fraudulent transactions in a **highly imbalanced credit card dataset** using machine learning techniques. The dataset contains anonymized features and transaction data, and the goal is to build an effective model to identify fraud while minimizing false positives and false negatives. Below is a detailed breakdown of the code and its functionality.

---

### **1. Data Loading and Exploration**
The project starts by loading the dataset and performing exploratory data analysis to understand its structure and key characteristics:
- **Loading Data**:
  The dataset (`creditcard.csv`) is loaded using **Pandas**, enabling quick manipulation and exploration.
- **Peeking at the Data**:
  - `.head()`: Displays the first few rows of the dataset.
  - `.describe()`: Provides summary statistics like mean, standard deviation, and feature ranges.
- **Output**:
  - The dataset has `284,807` rows and `31` columns (including the `Class` column, which identifies fraud).

---

### **2. Dataset Characteristics**
- **Imbalance in Data**:
  - The dataset is highly imbalanced, with only 0.17% of transactions being fraudulent.
  - Fraud cases (`Class=1`): 492  
    Valid transactions (`Class=0`): 284,315
  - An **outlier fraction** is computed to quantify the imbalance.

- **Insights**:
  - Fraudulent transactions tend to have higher average transaction amounts compared to valid transactions.
  - This imbalance makes it essential to handle the dataset carefully during model evaluation.

---

### **3. Correlation Analysis**
- **Correlation Matrix**:
  - A heatmap is plotted using **Seaborn** to visualize feature correlations.
  - Insights:
    - Most features have weak correlations with each other.
    - Certain features (e.g., V2, V5, and V20) show strong negative or positive correlations with the `Amount` column.

---

### **4. Feature and Target Separation**
- The data is divided into:
  - **Input features (`X`)**: All columns except `Class`.
  - **Target labels (`Y`)**: The `Class` column, indicating fraud or not.
- Output:
  - `X`: Shape `(284,807, 30)`
  - `Y`: Shape `(284,807, )`

---

### **5. Train-Test Split**
- The dataset is split into training (80%) and testing (20%) sets using `train_test_split` from **Scikit-learn**.
- This ensures that the model can be trained on one subset of the data and evaluated on unseen data.

---

### **6. Random Forest Classifier**
- A **Random Forest Classifier** is built using the `RandomForestClassifier` class from Scikit-learn.
- **Steps**:
  - The classifier is trained on the training set (`xTrain` and `yTrain`).
  - Predictions are made on the test set (`xTest`).
- The Random Forest model is chosen for its robustness and ability to handle imbalanced datasets effectively.

---

### **7. Evaluation Metrics**
The model is evaluated using the following metrics:
1. **Accuracy**:
   Measures the overall correctness of the predictions.
2. **Precision**:
   Indicates how many predicted fraud cases are truly fraud.
3. **Recall**:
   Measures the ability to detect fraudulent cases from all fraud cases.
4. **F1-Score**:
   A harmonic mean of precision and recall.
5. **Matthews Correlation Coefficient (MCC)**:
   A balanced measure even for imbalanced datasets.

**Example Results**:
- Accuracy: `99.96%`
- Precision: `98.67%`
- Recall: `75.51%`
- F1-Score: `85.55%`
- MCC: `0.8629`

---

### **8. Confusion Matrix Visualization**
- A confusion matrix is visualized to analyze true positives, true negatives, false positives, and false negatives.
- **Seaborn Heatmap**:
  - Rows represent actual classes (`Normal` vs. `Fraud`).
  - Columns represent predicted classes.
- Insights:
  - The model achieves high true positive and true negative rates but may still misclassify some fraud cases due to class imbalance.

---

### **9. Key Observations**
- Fraudulent transactions are rare but tend to have larger transaction amounts.
- The Random Forest model achieves excellent accuracy and precision but could further improve recall by addressing the data imbalance.
- Techniques like **SMOTE** (Synthetic Minority Oversampling Technique) or undersampling could help balance the dataset for better performance.

---

### **10. Future Improvements**
- Implement advanced techniques like **Gradient Boosting** (e.g., XGBoost, LightGBM) to improve performance.
- Explore methods to handle imbalanced data, such as:
  - Oversampling (e.g., SMOTE).
  - Undersampling the majority class.
- Perform hyperparameter tuning using Grid Search or Randomized Search for optimized model performance.

---

### **How to Run the Code**
1. Clone the repository and load the dataset (`creditcard.csv`).
2. Install the required libraries:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the script to train the Random Forest model and evaluate its performance.

---

This project demonstrates the end-to-end process of building a machine learning pipeline for fraud detection and provides actionable insights for further improvement.
