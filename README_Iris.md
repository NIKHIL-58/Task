
# Iris Flower Classification with Random Forest

![Iris](https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Iris_versicolor_2.jpg/220px-Iris_versicolor_2.jpg)

## Project Overview

This project demonstrates the application of a Random Forest Classifier to classify the **Iris** dataset into three species: Setosa, Versicolor, and Virginica. The goal is to predict the species of an iris flower based on its physical attributes: sepal length, sepal width, petal length, and petal width.

## Project Structure

- **Data Source**: Iris dataset from `sklearn.datasets`.
- **Libraries Used**:
  - `pandas`
  - `numpy`
  - `sklearn` (Random Forest, Evaluation Metrics)
  - `matplotlib` / `seaborn` for data visualization
  - `plotly.express` for interactive plots
  - `joblib` to save the trained model
  
## Features

### 1. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics and data cleaning.
   - Pairplot for visualizing relationships between features.
   - Correlation heatmap to examine feature correlations.

### 2. **Model Training**
   - Data splitting using `train_test_split`.
   - Feature scaling with `StandardScaler`.
   - Training a Random Forest Classifier (`RandomForestClassifier` from `sklearn`).
   
### 3. **Model Evaluation**
   - Accuracy score, classification report, and confusion matrix.
   - Precision-Recall and ROC curves to assess the model's performance.
   - Feature importance visualization to highlight important features.
   
### 4. **Precision-Recall Curve for Multiclass Classification**
   - Calculating precision-recall curves for each class using a one-vs-rest approach.

### 5. **Saving the Model**
   - The trained model is saved using `joblib` for future use.

## Requirements

The following libraries are required to run the code:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly joblib
```

## How to Run

1. Clone this repository or download the code.
2. Ensure all dependencies are installed.
3. Run the script. The following steps will be executed:
   - **Data Loading**: The Iris dataset will be loaded and preprocessed.
   - **Model Training**: The Random Forest model will be trained on the dataset.
   - **Model Evaluation**: Accuracy, classification report, confusion matrix, ROC, and precision-recall curves will be displayed.
   - **Model Saving**: The trained model will be saved to a file (`iris_rf_model.pkl`).

```bash
python iris_flower_classification.py
```

## Output

1. **Model Accuracy**: Displayed after model evaluation.
2. **Classification Report**: Provides precision, recall, and F1-score for each class.
3. **Confusion Matrix**: Visualized with a heatmap.
4. **Feature Importance**: Bar chart showing feature importance.
5. **ROC Curve**: Evaluates the model's true positive and false positive rate.
6. **Precision-Recall Curves**: Precision vs Recall curve for each class (Setosa, Versicolor, Virginica).
7. **Saved Model**: The trained model is saved in the `iris_rf_model.pkl` file.

## Example Output:

### Classification Report:
```text
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00         10
  Versicolor       0.90      0.90      0.90         10
   Virginica       0.90      0.90      0.90         10

    accuracy                           0.93         30
   macro avg       0.93      0.93      0.93         30
weighted avg       0.93      0.93      0.93         30
```

### Precision-Recall Curve:
- Displayed for each class, showing the trade-off between precision and recall.

## Conclusion

This project provides a hands-on example of how to perform classification on the Iris dataset using Random Forest, evaluate the model with various metrics, and visualize results with precision-recall and ROC curves. The trained model is saved for future predictions.

---

### 🚀 Future Improvements:
- Use different classification algorithms (e.g., Support Vector Machines, k-Nearest Neighbors).
- Perform hyperparameter tuning for Random Forest.
- Apply cross-validation for more robust model evaluation.
