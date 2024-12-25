
# Titanic Survival Prediction with Random Forest

This project involves building a machine learning model to predict the survival of passengers aboard the Titanic using the Titanic dataset. The model is built using a Random Forest classifier with advanced preprocessing, feature engineering, and hyperparameter tuning. This README provides an overview of the project, installation instructions, and how to run the code.

## 📊 Project Overview

In this project, we use the **Titanic dataset** from Kaggle to predict whether a passenger survived the Titanic disaster based on features such as:

- **Age**
- **Sex**
- **Fare**
- **FamilySize**
- **Embarked**
- **Pclass** (Ticket class)

### Key Features of the Project:
- Data preprocessing: Handling missing values and scaling numerical features
- Feature engineering: Creating new features like Family Size and IsAlone
- Model building: Using a **Random Forest classifier**
- Hyperparameter tuning: Using **GridSearchCV** to optimize model parameters
- Evaluation: Accuracy, Classification Report, and Confusion Matrix visualization
- Feature Importance visualization using a bar plot

## ⚙️ Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.6+
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `seaborn` - Data visualization
- `matplotlib` - Plotting
- `joblib` - Model serialization

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

## 📁 Project Files

1. **`titanic_survival_prediction.py`** – The main Python script containing all the code for data preprocessing, model training, evaluation, and hyperparameter tuning.
2. **`titanic_rf_model_advanced.pkl`** – The trained model saved after hyperparameter tuning.

## 📝 Code Explanation

### 1. **Data Loading**
The dataset is loaded from the following URL:  
[https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

```python
titanic_df = pd.read_csv(data_url)
```

### 2. **Preprocessing**
We handle missing values for both numerical and categorical columns. Numerical columns are imputed using **KNNImputer** and categorical columns with **SimpleImputer**.

```python
numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```

### 3. **Feature Engineering**
We create two new features:
- **FamilySize**: Sum of `SibSp` and `Parch`
- **IsAlone**: A binary indicator whether the passenger is alone (no family onboard)

```python
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df['IsAlone'] = 1
titanic_df['IsAlone'].loc[titanic_df['FamilySize'] > 0] = 0
```

### 4. **Model Training**
A **Random Forest** model is used for prediction, and we utilize **GridSearchCV** for hyperparameter tuning.

```python
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
```

### 5. **Evaluation**
The model’s performance is evaluated using accuracy, classification report, and a confusion matrix.

```python
print("
Model Accuracy:", accuracy)
```

### 6. **Saving the Model**
Once the best model is identified, it is saved using `joblib`.

```python
joblib.dump(best_model, 'titanic_rf_model_advanced.pkl')
```

## 📈 Visualizations

### Confusion Matrix

The confusion matrix is plotted to visualize the performance of the classification model.

![Confusion Matrix](https://user-images.githubusercontent.com/123456789/103100783-623a1b00-4607-11eb-9c4c-93e6b917db0e.png)

### Feature Importance

A bar plot showing the importance of each feature in predicting survival.

![Feature Importance](https://user-images.githubusercontent.com/123456789/103100782-623a1b00-4607-11eb-9b6d-7ac989f3ca39.png)

## 🚀 How to Run the Code

1. Clone the repository or download the script.
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python titanic_survival_prediction.py
   ```

The model will train, and you'll see the accuracy and other evaluation metrics printed in the console. The trained model will also be saved as `titanic_rf_model_advanced.pkl`.

## 🛠️ Hyperparameter Tuning

Hyperparameter tuning is performed using **GridSearchCV** for parameters like:
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of the tree
- `min_samples_split`: Minimum number of samples required to split a node

Best Hyperparameters:
```python
{'classifier__n_estimators': 100, 'classifier__max_depth': 10, 'classifier__min_samples_split': 10}
```

## 🔄 License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 📞 Contact

For more information or if you have any questions, feel free to reach out!

- **Email**: your-email@example.com
- **GitHub**: [your-github-profile](https://github.com/your-profile)
