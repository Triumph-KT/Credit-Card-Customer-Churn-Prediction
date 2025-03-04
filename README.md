# Credit Card Customer Churn Prediction Using Logistic Regression and Support Vector Machines (SVM)

Developed machine learning models to predict credit card customer churn for Thera Bank, identifying key drivers of attrition and enabling proactive customer retention strategies through detailed behavior analysis and classification techniques.

## Project Overview

This project addresses the challenge of predicting customer churn in credit card services to minimize revenue loss from departing customers. By analyzing customer demographics, activity patterns, and credit usage, predictive models were built to identify customers at high risk of leaving, supporting strategic interventions for retention.

## Dataset

The **Thera Bank churn dataset** contains:
- 10,127 records of customer demographic, behavioral, and financial data.
- Key features include:
  - Customer Age
  - Gender
  - Income Category
  - Education Level
  - Relationship Length with Bank
  - Product Usage (Total_Relationship_Count)
  - Activity Levels (Total_Trans_Ct, Months_Inactive_12_mon)
  - Credit Utilization (Avg_Utilization_Ratio)
  - Attrition Status (target variable: churn vs. retained)

## Objectives

- Predict whether a customer is likely to churn.
- Identify the key drivers of churn.
- Reduce misclassification of high-risk customers.
- Support the bank with actionable insights to prevent customer attrition.
- Provide a data-driven foundation for retention strategies.

## Methods

### Data Preprocessing:
- Removed irrelevant identifiers (e.g., Client ID).
- Imputed missing categorical data using **SimpleImputer** with most frequent strategy.
- Applied **OneHotEncoding** for categorical variables.
- Scaled numerical features using **MinMaxScaler** to support SVM model convergence.
- Split data into training and testing sets with stratification to preserve class balance.

### Model Development:
Developed and evaluated multiple classification models:

- **Logistic Regression**:
  - Baseline classifier achieving **~90% accuracy**, but limited recall (**~44%**) for identifying churned customers.
  - Applied threshold optimization to improve recall-precision trade-offs.

- **Support Vector Machine (Linear Kernel)**:
  - Delivered improved recall (**~55%**) with stable accuracy (~90%).
  - Reduced overfitting while enhancing generalization on unseen data.

- **Support Vector Machine (RBF Kernel)**:
  - Applied probability calibration and decision threshold optimization.
  - Enhanced balance between false positives and false negatives with a threshold of **0.31**.

### Evaluation:
- Assessed model performance using **accuracy**, **precision**, **recall**, **F1-score**, and **ROC curves**.
- Used **confusion matrices** to visualize misclassification of churned vs. retained customers.
- Prioritized **recall** as the key metric to maximize identification of high-risk customers.

## Results

- **Logistic Regression** achieved **~44% recall** for churned customers.
- **SVM (Linear Kernel)** improved recall to **~55%**, making it the preferred model for retention strategies.
- Key churn indicators identified:
  - High **Contacts_Count_12_mon** (frequent customer complaints or unresolved issues).
  - High **Months_Inactive_12_mon** (long inactivity periods).
  - Low **Total_Relationship_Count** (fewer bank products held).
  - High **Avg_Utilization_Ratio** (heavy credit usage).
  - Lower-income brackets and highly educated customers showed higher churn tendencies.

## Business/Scientific Impact

- Enabled proactive identification of at-risk customers, allowing Thera Bank to intervene before churn occurs.
- Recommended:
  - Prioritizing customer support for those with frequent unresolved contact events.
  - Offering targeted cross-sell campaigns to increase product adoption and reduce churn risk.
  - Re-engagement campaigns for inactive customers to reactivate usage.
  - Tailoring loyalty programs for highly educated, financially savvy customers who are more likely to leave for better alternatives.
- Provided a scalable churn prediction framework applicable to other financial service products.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- GridSearchCV

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit-card-churn-prediction.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Load and preprocess the dataset.
   - Train Logistic Regression and SVM models.
   - Evaluate model performance.
   - Analyze feature importance and churn drivers.
   - Visualize misclassification patterns with confusion matrices.

## Future Work

- Implement advanced models such as **XGBoost** or **Gradient Boosting** to further improve recall.
- Explore **SMOTE** and other oversampling techniques to address class imbalance.
- Develop real-time churn monitoring systems integrated into CRM platforms.
- Expand feature engineering with transaction histories, customer feedback, and sentiment analysis.
- Periodically retrain the model to reflect evolving customer behaviors and trends.
