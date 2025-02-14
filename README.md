# Heart Disease Prediction Project

This project aims to predict the presence of heart disease in patients using machine learning algorithms. The dataset used contains various medical features, and the target variable indicates whether a patient has heart disease (1) or not (0). Three machine learning algorithms—Logistic Regression, Random Forest, and Support Vector Machine (SVM)—were implemented and evaluated for their performance.

## Project Overview
- **Objective**: Predict the presence of heart disease based on patient medical data.
- **Dataset**: The dataset contains 303 records with 13 features and a target variable.
- **Algorithms Used**:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Support Vector Machine (SVM)
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report, and ROC Curve.

## Results
The performance of the algorithms on the test dataset is as follows:
- **Logistic Regression**: 80% accuracy
- **Random Forest**: 79% accuracy
- **SVM**: 82% accuracy

The Support Vector Machine (SVM) achieved the highest accuracy among the three algorithms.

## Key Features
1. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of the target variable.
   - Analyzed correlations between features using a heatmap.
   - Explored pairwise relationships between features using a pairplot.

2. **Data Preprocessing**:
   - Checked for missing values.
   - Split the dataset into training and testing sets (80-20 split).

3. **Model Training and Evaluation**:
   - Trained and evaluated three machine learning models.
   - Generated confusion matrices and classification reports for each model.
   - Plotted ROC curves to visualize model performance.

4. **Comparison of Algorithms**:
   - Compared the accuracy of all three algorithms using a bar plot.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Gayatridandgal/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install Dependencies**:
   Ensure you have the required Python libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Run the Notebook**:
   Open the Jupyter Notebook (`Heart_disease_prediction.ipynb`) and run the cells to reproduce the results.

4. **Explore the Results**:
   - Check the accuracy scores, confusion matrices, and ROC curves for each algorithm.
   - Compare the performance of the three algorithms.

## Files in the Repository
- `Heart_disease_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `heart_disease_data.csv`: Dataset used for the project.
- `README.md`: This file, providing an overview of the project.
  
Happy Coding!
