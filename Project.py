import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Added for folder operations

# Create Visualization folder if it doesn't exist
visualization_folder = 'Visualization'
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)
    print(f"Created folder: {visualization_folder}")
else:
    print(f"Folder already exists: {visualization_folder}")

# Load the dataset
df = pd.read_csv("Diabetes_Project/Dataset_Diabetes.csv")

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Step 1: Understanding the dataset - Basic exploration
print(df.info())

# Step 2: Check for null values - Data quality assessment
print("Null values:")
print(df.isnull().sum())

# Step 3: Check for duplicates - Data quality assessment
print(f"Duplicated rows: {df.duplicated().sum()}")

# Step 4: Categorical Columns (Class & Gender) - Identify non-numeric data
cat_col = df.select_dtypes(include='object').columns
print("Categorical columns:", cat_col)

# Display unique values for each categorical column to understand data distribution
for col in cat_col:
    print(f"{col} unique values: {df[col].unique()}")

# Data cleaning: Standardize inconsistent values in categorical columns
df['Gender'] = df['Gender'].replace({'f':'F'})  # Standardize gender notation
df['CLASS'] = df['CLASS'].replace({'N ':'N','Y ':'Y'})  # Remove trailing spaces

print("After cleaning:")
for col in cat_col:
    print(f"{col} unique values: {df[col].unique()}")

# Feature Engineering: Convert categorical variables to numerical format
gender_encoder = LabelEncoder()
df['Gender_encoded'] = gender_encoder.fit_transform(df['Gender'])  # Convert Gender to 0/1
df = pd.get_dummies(df, columns=['CLASS'], prefix='Class', dtype=int)  # One-hot encode CLASS

# Feature Selection: Remove irrelevant columns that could distort analysis
num_col = df.select_dtypes(exclude='object').columns.drop(['ID', 'No_Pation','Gender_encoded','Class_N','Class_P','Class_Y'])
print("Numerical columns for analysis:", num_col)

# Outlier Detection: Identify extreme values using IQR method
q1 = df[num_col].quantile(0.25)  # First quartile (25th percentile)
q3 = df[num_col].quantile(0.75)  # Third quartile (75th percentile)
iqr = q3 - q1  # Interquartile range

# Create boolean mask for outliers (values outside 1.5*IQR from quartiles)
outlier_mask = (df[num_col] < (q1 - 1.5 * iqr)) | (df[num_col] > (q3 + 1.5 * iqr))
outlier_counts = outlier_mask.sum()
print("Outlier counts:")
print(outlier_counts)

# Visualizing the outliers before treatment - Boxplot to see data distribution
df[num_col].boxplot(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Outliers Before Treatment")
plt.tight_layout()
plt.savefig(f'{visualization_folder}/outliers_before_treatment.png', dpi=300, bbox_inches='tight')
plt.show()

# Outlier Treatment: Replace outliers with NaN for imputation
df_masked = df.copy()
df_masked.loc[:, num_col] = df_masked.loc[:, num_col].astype('float64')  # Ensure numeric type
df_masked.loc[:, num_col] = df_masked.loc[:, num_col].mask(outlier_mask)  # Mask outliers as NaN

# Data Imputation: Fill missing values (from outlier removal) with median
imputer = SimpleImputer(strategy='median')  # Use median for robustness to outliers
imputed_values = imputer.fit_transform(df_masked[num_col])

df_cleaned = df_masked.copy()
df_cleaned.loc[:, num_col] = imputed_values  # Replace with imputed values

# Feature Scaling: Normalize all numerical features to [0,1] range
scaler = MinMaxScaler()
df_cleaned[num_col] = scaler.fit_transform(df_cleaned[num_col])  # Scale features

print("After scaling:")
print(df_cleaned[num_col].describe())

# Visualizing after outlier treatment and scaling - Compare with previous boxplot
df_cleaned[num_col].boxplot(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("After Outlier Treatment and Scaling")
plt.tight_layout()
plt.savefig(f'{visualization_folder}/after_outlier_treatment_scaling.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# NEW ADDITIONS FOR VISUALIZATION AND ANALYSIS
###############################################################################

# Create correlation matrix with output variables
print("\n=== Creating Correlation Heatmap with Output Variables ===")

# Select only the features (numerical columns) and output variables
feature_cols = num_col.tolist()
output_cols = ['Class_N', 'Class_P', 'Class_Y']  # Your one-hot encoded output variables

# Create a combined dataset for correlation analysis
correlation_data = pd.concat([df_cleaned[feature_cols], df_cleaned[output_cols]], axis=1)

# Calculate correlation matrix using Spearman (handles non-linear relationships)
correlation_matrix = correlation_data.corr(method='spearman')

# Extract only the correlations between features and outputs (the part we care about)
feature_output_corr = correlation_matrix.loc[feature_cols, output_cols]

print("Feature-Output Correlation Matrix:")
print(feature_output_corr.round(3))

# Create heatmap focusing on feature-output correlations
plt.figure(figsize=(12, 8))
sns.heatmap(feature_output_corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.3f',
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10})

plt.title('Feature-Output Correlation Heatmap (Spearman)\nDiabetes Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Output Classes (N=Negative, P=Positive, Y=Prediabetes)')
plt.ylabel('Features')
plt.tight_layout()

# Save the heatmap to Visualization folder
plt.savefig(f'{visualization_folder}/feature_output_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create the full correlation heatmap for completeness
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})

plt.title('Full Correlation Heatmap - Features and Outputs (Spearman)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the full correlation heatmap
plt.savefig(f'{visualization_folder}/full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the strongest correlations (both positive and negative)
print("\n=== Top Feature-Output Correlations ===")
corr_values = []
for feature in feature_cols:
    for output in output_cols:
        corr_value = feature_output_corr.loc[feature, output]
        corr_values.append((feature, output, corr_value))

# Sort by absolute correlation value to find most influential features
corr_values.sort(key=lambda x: abs(x[2]), reverse=True)

print("Top correlations (absolute value):")
for feature, output, corr in corr_values[:10]:  # Top 10
    print(f"{feature} - {output}: {corr:.3f}")

# Check if any correlations are above 0.7 (strong correlation threshold)
strong_correlations = [(f, o, c) for f, o, c in corr_values if abs(c) > 0.7]
if strong_correlations:
    print(f"\nStrong correlations (|r| > 0.7): {len(strong_correlations)} found")
    for feature, output, corr in strong_correlations:
        print(f"{feature} - {output}: {corr:.3f}")
else:
    print(f"\nNo strong correlations found (|r| > 0.7)")
    print("Highest correlation:", f"{corr_values[0][0]} - {corr_values[0][1]}: {corr_values[0][2]:.3f}")

# Distribution plots to visualize scaled feature distributions
print("\nCreating distribution plots...")
df_cleaned[feature_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Scaled Features", fontsize=16)
plt.tight_layout()

# Save distribution plots
plt.savefig(f'{visualization_folder}/scaled_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the cleaned dataset to Visualization folder
df_cleaned.to_csv(f'{visualization_folder}/diabetes_cleaned_dataset.csv', index=False)
print(f"\nCleaned dataset saved to '{visualization_folder}/diabetes_cleaned_dataset.csv'")

# Create and save the correlation matrix as CSV
feature_output_corr.to_csv(f'{visualization_folder}/feature_output_correlations.csv')
print(f"Feature-output correlations saved to '{visualization_folder}/feature_output_correlations.csv'")

# Save the full correlation matrix as CSV
correlation_matrix.to_csv(f'{visualization_folder}/full_correlation_matrix.csv')
print(f"Full correlation matrix saved to '{visualization_folder}/full_correlation_matrix.csv'")

print(f"\n=== All files saved successfully in '{visualization_folder}' folder ===")
print("Generated files:")
generated_files = os.listdir(visualization_folder)
for file in generated_files:
    print(f"  - {file}")