# Titanic Survival Prediction

This project predicts the survival of passengers on the Titanic using a Random Forest Classifier. The dataset is sourced from the famous Titanic dataset available [here](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv).

## Dataset
The dataset contains information about Titanic passengers, including features such as:
- **Age**
- **Gender (Sex)**
- **Ticket Class (Pclass)**
- **Embarkation Port (Embarked)**

## Steps in the Project
1. **Data Preprocessing**:
   - Handled missing values.
   - Encoded categorical features (e.g., Sex, Embarked).
   - Dropped irrelevant columns (e.g., PassengerId, Name, Ticket).
2. **Feature Scaling**:
   - Standardized numerical features for improved model performance.
3. **Model Selection**:
   - Used a Random Forest Classifier.
4. **Evaluation**:
   - Achieved high accuracy on the test data.
   - Generated a classification report and confusion matrix.

## Results
- **Accuracy**: Achieved over 80% accuracy on the test data.
- **Feature Importance**: Gender (Sex) and Ticket Class (Pclass) were identified as important features.

## How to Run
1. Clone the repository.
2. Install the required libraries: `pip install -r requirements.txt`.
3. Run the Python script: `python titanic_survival_prediction.py`.

## Dependencies
- Python 3.7+
- scikit-learn
- pandas
- seaborn
- matplotlib

"""
