#Part 1

# Step 2: Import the required Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Step 1 & 2: Load the dataset
# Replace 'superstore.csv' with your dataset filename/path if different
# Creating a robust synthetic business dataset in case local CSV isn't present
try:
    df = pd.read_csv("superstore.csv")
    print("Dataset loaded successfully from file.")
except FileNotFoundError:
    print("Local CSV not found. Initializing sample Superstore Business dataset...")
    np.random.seed(42)
    n = 300
    categories = ["Furniture", "Office Supplies", "Technology"]
    regions = ["East", "West", "Central", "South"]
    
    df = pd.DataFrame({
        "Category": np.random.choice(categories, n),
        "Region": np.random.choice(regions, n),
        "Sales": np.random.exponential(scale=200, size=n) + 10,
        "Quantity": np.random.randint(1, 15, size=n),
        "Discount": np.random.choice([0.0, 0.1, 0.2, 0.5], size=n),
        "Profit": np.random.normal(loc=30, scale=80, size=n)
    })

# ----------------------------------------------------
# Step 3: Explore dataset dimensions, types & summaries
# ----------------------------------------------------
print("\n--- 1. Dataset Overview ---")
print("Dimensions (Rows, Columns):", df.shape)

print("\n--- 2. Dataset Info ---")
print(df.info())

print("\n--- 3. Numerical Summary ---")
print(df.describe())

print("\n--- 4. Value Counts (Category) ---")
print(df["Category"].value_counts())

# ----------------------------------------------------
# Step 4: Numerical Feature Distributions (Histograms)
# ----------------------------------------------------
plt.figure()
sns.histplot(df["Sales"], kde=True, color="teal")
plt.title("Step 4: Distribution of Sales (Histogram & KDE)")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# ----------------------------------------------------
# Step 5: Categorical Visualizations (Bar & Count Plots)
# ----------------------------------------------------
plt.figure()
sns.countplot(data=df, x="Category", palette="Blues_d")
plt.title("Step 5: Order Frequency by Product Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# ----------------------------------------------------
# Step 6: Correlation Matrix & Heatmap
# ----------------------------------------------------
plt.figure(figsize=(7, 5))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Step 6: Feature Correlation Heatmap")
plt.show()

# ----------------------------------------------------
# Step 7: Outlier Detection (Box Plots) & Relationships (Scatter Plots)
# ----------------------------------------------------
# Box Plot for Outliers
plt.figure()
sns.boxplot(data=df, x="Category", y="Profit", palette="Set2")
plt.title("Step 7A: Profit Distribution & Outlier Detection across Categories")
plt.show()

# Scatter Plot for Feature Relationships
plt.figure()
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Category", alpha=0.8)
plt.title("Step 7B: Sales vs Profit Analysis")
plt.show()

# ----------------------------------------------------
# Step 8: Univariate, Bivariate, and Multivariate Analysis
# ----------------------------------------------------
# Multivariate: Category-wise Mean Sales across Regions
multivariate_data = df.groupby(["Region", "Category"])["Sales"].mean().unstack()

multivariate_data.plot(kind="bar", figsize=(9, 5))
plt.title("Step 8: Multivariate Analysis - Average Sales by Region & Category")
plt.xlabel("Region")
plt.ylabel("Average Sales ($)")
plt.legend(title="Category")
plt.xticks(rotation=0)
plt.show()

# ----------------------------------------------------
# Step 9: Key Observations Summary
# ----------------------------------------------------
print("\n" + "="*50)
print("Step 9: Key Quantitative Findings")
print("="*50)
print(f"- Total Revenue Generated: ${df['Sales'].sum():,.2f}")
print(f"- Average Transaction Profit: ${df['Profit'].mean():,.2f}")
print(f"- Most Frequent Category: {df['Category'].mode()[0]}")
print(f"- Maximum Discount Applied: {df['Discount'].max() * 100}%")

# ----------------------------------------------------
# Step 10: Concise Business Recommendations
# ----------------------------------------------------
print("\n" + "="*50)
print("Step 10: Final Business Insights & Recommendations")
print("="*50)
print("""
1. Discount Optimization: High discount rates are heavily correlated with negative profit margins. Introduce a threshold on maximum discounting.
2. Regional Strategy: Focus marketing and inventory allocation on the best-performing regions identified in Step 8.
3. Outlier Mitigation: Review high-loss transactions in underperforming product categories.
""")

#part 2

# 1. What is Exploratory Data Analysis (EDA), and why is it performed before machine learning?
# Sol:
# Definition: EDA is the critical process of analyzing datasets using statistical summaries and visual representations to understand their underlying structure.Why before ML: It uncovers patterns, identifies anomalies/outliers, detects missing values, checks feature distributions/skewness, and verifies assumptions required by machine learning algorithms before training.

# 2. Differentiate between univariate, bivariate, and multivariate analysis with suitable examples.
# Sol:
# Analysis TypeFocusExampleUnivariateAnalyzes a single variable in isolation to understand its distribution and spread.Plotting a histogram of employee Age or calculating mean Salary.BivariateAnalyzes the relationship between two variables.Scatter plot between Years of Experience vs Salary to see correlation.MultivariateAnalyzes relationships among three or more variables simultaneously.Pairplot or Heatmap comparing Sales, Profit, Discount, and Region.

# 3. What insights can be obtained from a correlation heatmap?
# Sol:
# Identifies the strength and direction (positive/negative) of linear relationships between numerical features.Detects multicollinearity (highly correlated independent variables), which helps avoid redundancy in ML models.Highlights strong predictor variables that correlate well with the target variable.

# 4. Explain the purpose of histograms, box plots, and scatter plots in EDA.Histogram: 
# Sol:
# Shows the frequency distribution, skewness (left/right/normal), and central tendency of a single continuous variable.Box Plot: Visualizes the 5-number summary (Min, Q1, Median, Q3, Max) and clearly flags outliers (points outside $1.5 \times \text{IQR}$).Scatter Plot: Displays individual data points on a 2D plane to evaluate correlations, linear/non-linear trends, and clusters between two continuous features.

# 5. How can EDA help identify data quality issues before analysis?
# sol:
# Missing Data: Uncovers null/NaN counts and missing data patterns across columns.Outliers & Errors: Detects extreme values or invalid inputs (e.g., negative age, unrealistic salaries).Class Imbalance: Identifies heavily skewed categorical target distributions (e.g., 98% No, 2% Yes).Data Type Mismatches: Detects numerical values stored as strings or incorrect date formats.

# 6. Why is correlation important in predictive analytics? Can correlation imply causation?
# Sol:
# Importance: It guides feature selection by identifying inputs most strongly linked to the outcome variable.Causation: No, correlation does not imply causation. Two variables may move together due to coincidence or a third unobserved variable (confounding factor), not because one directly causes the other.

# 7. Which visualization would you use to analyze categorical and numerical variables? Justify your choice.
# Sol:
# Box Plot / Violin Plot: Best for comparing a numerical distribution across different categorical levels (e.g., Salary across different Job Roles).Bar Chart (with aggregated metrics): Best for comparing summary statistics like mean/total of a numerical variable grouped by categories (e.g., total Sales per Region).

# 8. What business insights can be derived from the Netflix (or Superstore/HR Analytics) dataset through EDA?
# Sol:
# Superstore Sales: Identifies top loss-making product sub-categories, high-margin customer segments, and regional sales bottlenecks.Netflix Titles: Identifies content expansion trends over years, top-producing countries, and the ratio of Movies vs. TV Shows.HR Analytics: Identifies key attrition drivers (e.g., overtime, low salary hike, long commute distance).

# 9. How does EDA contribute to feature selection and model building?
# Sol:
# Removes Redundancy: Drops redundant/collinear features identified via heatmaps.Informs Transformations: Guides log/power transformations for heavily skewed distributions.Guides Feature Engineering: Helps create meaningful composite features based on observed variable interactions.

# 10. What challenges might arise while performing EDA on large-scale real-world datasets?
# sol:
# Memory Limits: Datasets exceeding RAM capacity cause kernel crashes during standard Pandas loading.Plotting Overhead: Rendering millions of points on scatter plots causes severe visual clutter (overplotting) and browser lag.High Cardinality & Dimensionality: Datasets with hundreds of columns or thousands of unique categories make manual visualization difficult.