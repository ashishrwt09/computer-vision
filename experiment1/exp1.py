#Part-1

# Step 2: Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Step 1 & 3: Load the dataset and inspect structure
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

# Step 4: Identify and handle missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill numerical missing values with median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Fill categorical missing values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop column with excessive missing data
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Step 5: Detect and remove duplicate records
print(f"\nDuplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Step 6: Encode categorical variables
# Label Encoding for binary features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male: 1, female: 0

# One-Hot Encoding for multi-class categorical features
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 7: Outlier Detection and Treatment (IQR Method for Fare)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df['Fare'] = np.clip(df['Fare'], lower_bound, upper_bound)

# Step 8: Normalization / Standardization
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 9: Feature Engineering
# Create FamilySize = SibSp + Parch + 1 (self)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create AgeGroup category
df['AgeGroup'] = pd.cut(df['Age'], bins=3, labels=['Child', 'Adult', 'Senior'])

# Step 10: Save cleaned dataset
df.to_csv('titanic_cleaned.csv', index=False)
print("\nPreprocessing complete. Saved to 'titanic_cleaned.csv'.")


#Part-2

# 1. Why is data preprocessing considered one of the most important phases in data analytics?
# Sol:
# Raw data is often noisy, missing values, duplicated, and inconsistent. Preprocessing cleans and structures data so machine learning algorithms can converge efficiently and produce accurate, unbiased predictions without "garbage in, garbage out" failure.
# 2. Explain different methods of handling missing values with suitable examples.
# Sol: 
# Deletion: Dropping rows (df.dropna()) when missing entries are rare (<5%), or dropping columns (df.drop()) if most entries are missing (e.g., Cabin column).
# Mean/Median Imputation: Replacing missing numbers with the central tendency (e.g., filling missing Age with median $28$ to avoid skew).
# Mode Imputation: Replacing missing categorical values with the most frequent value (e.g., filling missing Embarked with 'S').
# Predictive/KNN Imputation: Estimating missing values based on similarities to other feature rows.

# 3. Differentiate between Label Encoding and One-Hot Encoding.
# Sol:
# FeatureLabel EncodingOne-Hot EncodingMechanismAssigns an integer to each category ($0, 1, 2...$)Creates separate binary columns ($0$ or $1$) for each unique valueBest Used ForOrdinal variables with inherent order (e.g., Low, Medium, High)Nominal variables without ordering (e.g., City: Delhi, Mumbai, Pune)RiskModels may misinterpret numerical order as mathematical weightIncreases dimensionality significantly (Curse of Dimensionality)

# 4. What are outliers? How can they affect analytical results?
# Sol:
# Outliers are data points that deviate significantly from the overall pattern of the data. They skew summary statistics (inflating the mean and standard deviation) and degrade the performance of sensitive models (such as Linear Regression, K-Means clustering, and Neural Networks).

# 5. Explain the difference between normalization and standardization.
# Sol: 
# Normalization (Min-Max Scaling): Rescales values to a fixed range, usually $[0, 1]$, using:$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$Useful for algorithms requiring bounded inputs (e.g., Neural Networks, KNN).Standardization (Z-Score Normalization): Centers data around a mean of $0$ with a standard deviation of $1$ using:$$Z = \frac{X - \mu}{\sigma}$$Less sensitive to outliers and standard for algorithms assuming Gaussian distributions (e.g., Logistic Regression, SVM, PCA).

# 6. Why should duplicate records be removed before analysis?
# Sol:
# Duplicate records lead to data leakage, falsely amplify patterns, skew model weights toward overrepresented points, and result in overoptimistic/inaccurate evaluation metrics.

# 7. What is feature engineering? Give two practical examples.
# Sol: 
# Feature engineering is the process of using domain knowledge to extract new variables from raw data to improve model predictive power.Example 1: Combining SibSp (siblings/spouses) and Parch (parents/children) into a single FamilySize metric.Example 2: Extracting DayOfWeek, Hour, or IsWeekend from a single Timestamp column.

# 8. Which preprocessing techniques would you apply to the IBM HR Employee Attrition dataset and why?
# SOl: 
# Removal of Zero-Variance Columns: Drop columns with a single unique value (like EmployeeCount, StandardHours, Over18) as they provide zero information.ID Removal: Drop identifier columns (EmployeeNumber) to prevent overfitting.One-Hot / Binary Encoding: Convert categorical attributes (OverTime, Department, JobRole, BusinessTravel) to numeric vectors.Feature Scaling: Apply StandardScaler to features like MonthlyIncome, TotalWorkingYears, and Age because they exist on vastly different numerical scales.Class Imbalance Handling: Handle the imbalanced target class (Attrition yes/no) using techniques like SMOTE or class weighting.

# 9. How does poor-quality data affect machine learning model performance?
# Sol:
# It causes biased estimates, high variance, poor generalizability on unseen data, slow convergence during training, and misleading feature importance rankings.

# 10. Name any three Python libraries commonly used for data preprocessing.
# Sol:
# Pandas (Data manipulation and cleaning)NumPy (Numerical operations and array handling)Scikit-Learn (sklearn.preprocessing module for scaling, encoding, and imputation)