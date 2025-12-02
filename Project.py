import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from apyori import apriori
import pyfpgrowth
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


df=pd.read_csv("Dataset_Diabetes.csv")

#Step 1: Understanding the dataset
#print(df.head())
print(df.info())
#print(df.describe())

#Step2: Null
#check for null values even though df.info confirmed there arent any
print(df.isnull().sum())
#Step3: Duplicates
#check for duplicated rows
print(df.duplicated().sum())

#Step4: Categorical Columns (Class & Gender)
cat_col=df.select_dtypes(include = 'object').columns
print (cat_col)

for col in cat_col:
    print(df[col].unique())

df['Gender'] = df['Gender'].replace({'f':'F'})
df['CLASS'] = df['CLASS'].replace({'N ':'N','Y ':'Y'})

for col in cat_col:
    print(df[col].unique())

#now we have consistent values
#label encode Gender
gender_encoder= LabelEncoder()
df['Gender_encoded']=gender_encoder.fit_transform(df['Gender'])
#One hot encode Class
df['CLASS_original'] = df['CLASS']  # Save before one-hot encoding
df = pd.get_dummies(df, columns=['CLASS'], prefix='Class', dtype=int)

#print(df)

#drop id and number of patients because they will distort the results
num_col=df.select_dtypes(exclude='object').columns.drop(['ID', 'No_Pation','Gender_encoded','Class_N','Class_P','Class_Y'])
print (num_col)

#now we check for outliers
q1= df[num_col].quantile(0.25)
q3 = df[num_col].quantile(0.75)
iqr= q3-q1

outlier_mask = (df[num_col] < (q1 - 1.5 * iqr)) | (df[num_col] > (q3 + 1.5 * iqr))

outlier_counts = outlier_mask.sum()
print(outlier_counts)


#visualizing the outliers

# Get global min and max from the original (before cleaning) DataFrame
ymin = df[num_col].min().min() -5
ymax = df[num_col].max().max() + 5

# --- Before removing outliers ---
plt.figure(figsize=(12,6))
df[num_col].boxplot()
plt.title("Before Outlier Removal")
plt.ylim(ymin, ymax)  # fix y scale
plt.xticks(rotation=45)
plt.savefig("visualization/Boxplot_before_outlier_removal.png", dpi=300, bbox_inches='tight')
plt.show()

#df_cleaned = df[~outlier_mask.any(axis=1)]

#mask outlier cells with NaN
df_masked = df.copy()
df_masked.loc[:, num_col] = df_masked.loc[:, num_col].astype('float64')
df_masked.loc[:, num_col] = df_masked.loc[:, num_col].mask(outlier_mask)

#impute missing values (from outlier masking) with median
imputer = SimpleImputer(strategy='median')
imputed_values = imputer.fit_transform(df_masked[num_col])

df_cleaned = df_masked.copy()
df_cleaned.loc[:, num_col] = imputed_values


#######################################EVALUTAING OUTLIER MASKING AND IMPUTATION#######################################

#how many outliers were there
total_outliers = outlier_mask.sum().sum()
print(f"\nTotal outlier cells detected: {total_outliers}")
print(outlier_mask.sum().sort_values(ascending=False))


#how many NaNs introduced before imputation
print("\nNaNs introduced (before imputation):")
print(df_masked[num_col].isna().sum().sort_values(ascending=False))

#Check that imputation removed all NaNs
print("\nRemaining NaNs after imputation (should be 0):")
print(df_cleaned[num_col].isna().sum().sum())

# Check medians didn’t shift wildly
print(pd.DataFrame({
    'Median_before': df[num_col].median().round(3),
    'Median_after' : df_cleaned[num_col].median().round(3)
}))

#check means didn’t shift wildly
print(pd.DataFrame({
    'Mean_before': df[num_col].mean().round(3),
    'Mean_after' : df_cleaned[num_col].mean().round(3)
}))


#######################################EVALUTAING OUTLIER MASKING AND IMPUTATION#######################################


#######################################Scatter plots before removing outliers#######################################
plt.figure(figsize=(10, 5))
for dclass, color, marker in zip(['N', 'P', 'Y'], ['blue', 'orange', 'red'], ['o', 's', '^']):
    subset = df[df['CLASS_original'] == dclass]
    plt.scatter(subset['AGE'], subset['BMI'],
                alpha=0.6, c=color, label=f'Class {dclass}', marker=marker, s=50)

plt.xlabel('Age', fontsize=12)
plt.ylabel('BMI', fontsize=12)
plt.title('Age vs BMI before removing outliers', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("visualization/Age_vs_BMI_before_outliers.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
for dclass, color, marker in zip(['N', 'P', 'Y'], ['blue', 'orange', 'red'], ['o', 's', '^']):
    subset = df[df['CLASS_original'] == dclass]
    plt.scatter(subset['BMI'], subset['HbA1c'],
                alpha=0.6, c=color, label=f'Class {dclass}', marker=marker, s=50)

plt.xlabel('BMI', fontsize=12)
plt.ylabel('HbA1c', fontsize=12)
plt.title('BMI vs HbA1c before removing outliers', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("visualization/BMI_vs_HbA1c_before_outliers.png", dpi=300, bbox_inches='tight')
plt.show()


#from the plot we can see that the Y class (diabetes= yes) often has BMI>25 and Age>50

#######################################Scatter plots before removing outliers#######################################


#######################################Scatter plots after removing outliers#######################################

#visualizing after removing outliers
plt.figure(figsize=(12,6))
df_cleaned[num_col].boxplot()
plt.title("Boxplot After Outlier Imputation (Before Scaling)")
plt.ylim(ymin, ymax)  # same y scale as 'before'
plt.xticks(rotation=45)
plt.savefig("visualization/Boxplot_after_outlier_imputation.png", dpi=300, bbox_inches='tight')
plt.show()

# Subplots for each numeric column
df_cleaned[num_col].plot.box(
    subplots=True,
    layout=(2, 5),          # 2 rows x 5 columns grid
    figsize=(14, 6),
    sharey=False,
    title='Boxplots After Outlier Imputation (Per Feature)'
)
plt.tight_layout()
plt.savefig("visualization/Boxplot_after_outlier_imputation_subplots.png", dpi=300, bbox_inches='tight')
plt.show()


#Scatter plots after removing outliers
# Age vs BMI
plt.figure(figsize=(10, 5))
for dclass, color, marker in zip(['N', 'P', 'Y'], ['blue', 'orange', 'red'], ['o', 's', '^']):
    subset = df_cleaned[df_cleaned['CLASS_original'] == dclass]
    plt.scatter(subset['AGE'], subset['BMI'], 
                alpha=0.6,           # Transparency (0-1) to see overlapping points
                c=color,             # Color for this class
                label=f'Class {dclass}',
                marker=marker,       # Different shapes for each class
                s=50)               # Size of markers

plt.xlabel('Age', fontsize=12)
plt.ylabel('BMI', fontsize=12)
plt.title('Age vs BMI after removing outliers', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)  # Light grid
plt.savefig("visualization/Age_vs_BMI_after_outliers.png", dpi=300, bbox_inches='tight')
plt.show()

# Age vs HbA1c
plt.figure(figsize=(10, 6))
for dclass, color, marker in zip(['N', 'P', 'Y'], ['blue', 'orange', 'red'], ['o', 's', '^']):
    subset = df_cleaned[df_cleaned['CLASS_original'] == dclass]
    plt.scatter(subset['BMI'], subset['HbA1c'], 
                alpha=0.6, c=color, label=f'Class {dclass}', 
                marker=marker, s=50)

plt.xlabel('BMI', fontsize=12)
plt.ylabel('HbA1c', fontsize=12)
plt.title('BMI vs HbA1c after removing outliers', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("visualization/BMI_vs_HbA1c_after_outliers.png", dpi=300, bbox_inches='tight')
plt.show()

#from the plot we can see that the Y class (diabetes= yes) often has BMI>25 and Age>50

#######################################Scatter plots after removing outliers#######################################

#now scaling
scaler=MinMaxScaler()
df_cleaned[num_col] = scaler.fit_transform(df_cleaned[num_col])
df_cleaned[num_col].describe()

df_cleaned[num_col].hist(bins=20, figsize=(15,10))
plt.suptitle("Histogram of Scaled Features")
plt.savefig("visualization/Histogram_scaled_features.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df_cleaned[num_col].corr(method='spearman'), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Scaled Features")
plt.savefig("visualization/Correlation_heatmap_scaled_features.png", dpi=300, bbox_inches='tight')
plt.show()


#data splitting
# Target column (the original version before one-hot encoding)
y = df_cleaned['CLASS_original']

# Feature set (numerical + encoded gender)
X = df_cleaned[num_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#######################################Decision Tree#######################################



print("\n" + "="*60)
print("DECISION TREE IMPLEMENTATION")
print("="*60)

# Create and train decision tree
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_leaf=10,
    class_weight={'N': 3, 'P': 2, 'Y': 1}
)


clf.fit(X_train, y_train)

# Display the tree
plt.figure(figsize=(20, 12))
plot_tree(clf,
          feature_names=X_train.columns.tolist(),
          class_names=sorted(y_train.unique()),
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Diabetes Classification", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("visualization/decision_tree_structure.png", dpi=300, bbox_inches='tight')
plt.show()

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Display sample predictions
print("\nFirst 10 predictions (Actual -> Predicted):")
for i in range(min(10, len(y_test))):
    print(f"  {y_test.iloc[i]} -> {y_pred[i]}")

print("\n" + "="*60)
print("DETAILED EVALUATION METRICS")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['N', 'P', 'Y'])

# Debug: Check actual confusion matrix values
print(f"\n=== DEBUG: Actual Confusion Matrix Values ===")
print(f"Test set size: {len(y_test)}")
print(f"Confusion matrix shape: {cm.shape}")
print(f"Confusion matrix:\n{cm}")
print(f"Class distribution in test set:")
print(y_test.value_counts().sort_index())

print("\n" + "="*50)
print("CONFUSION MATRIX (Formatted)")
print("="*50)

# Display formatted confusion matrix
print("\n                Predicted")
print("               ┌─────┬─────┬─────┐")
print("               │  N  │  P  │  Y  │")
print("               ├─────┼─────┼─────┤")

# Print rows with proper formatting
for i, true_label in enumerate(['N', 'P', 'Y']):
    if i == 0:
        print(f"Actual   N: │ {cm[0,0]:3d} │ {cm[0,1]:3d} │ {cm[0,2]:3d} │")
    elif i == 1:
        print(f"Actual   P: │ {cm[1,0]:3d} │ {cm[1,1]:3d} │ {cm[1,2]:3d} │")
    else:
        print(f"Actual   Y: │ {cm[2,0]:3d} │ {cm[2,1]:3d} │ {cm[2,2]:3d} │")
        print("               └─────┴─────┴─────┘")

print("\nKey:")
print("N = Normal, P = Prediabetes, Y = Diabetes")
print("Rows = True/Actual class")
print("Columns = Predicted class")

# Calculate accuracy manually from confusion matrix
total = cm.sum()
correct = np.diag(cm).sum()
accuracy_manual = correct / total

print(f"\nFrom confusion matrix:")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")

# Verify accuracy matches
print(f"\nVerification: accuracy_score = {accuracy:.4f}, manual calculation = {accuracy_manual:.4f}")
print(f"Match: {abs(accuracy - accuracy_manual) < 0.0001}")

# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
report = classification_report(y_test, y_pred,
                               target_names=['N', 'P', 'Y'],
                               output_dict=True)
print(classification_report(y_test, y_pred, target_names=['N', 'P', 'Y']))

# Extract precision and recall for each class
print("\n=== Detailed Per-Class Metrics ===")
for class_name in ['N', 'P', 'Y']:
    precision = report[class_name]['precision']
    recall = report[class_name]['recall']
    f1 = report[class_name]['f1-score']
    support = report[class_name]['support']

    print(f"\nClass {class_name}:")
    print(f"  Precision: {precision:.4f} - Of predicted {class_name}, {precision*100:.1f}% were correct")
    print(f"  Recall:    {recall:.4f} - Found {recall*100:.1f}% of actual {class_name} cases")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Support:   {support} samples")

# Visualize confusion matrix
plt.figure(figsize=(10, 8))

# Create heatmap with actual values
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=['N', 'P', 'Y'],
                 yticklabels=['N', 'P', 'Y'],
                 cbar_kws={'label': 'Number of Cases'},
                 annot_kws={"size": 14, "weight": "bold"})

# Customize title and labels
plt.title(f'Decision Tree Confusion Matrix\nAccuracy: {accuracy*100:.2f}%',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

# Add custom annotations for errors
for i in range(3):
    for j in range(3):
        if i != j and cm[i, j] > 0:  # Highlight misclassifications
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}',
                    ha='center', va='center',
                    fontweight='bold', color='red',
                    fontsize=14)

plt.tight_layout()
plt.savefig("visualization/decision_tree_confusion_matrix_ACTUAL.png",
            dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("ADDITIONAL METRICS CALCULATION")
print("="*60)

# Calculate metrics for each class
print("\n=== Detailed Metrics Calculation ===")

# For Class N
tn_n = cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]
fp_n = cm[0,1] + cm[0,2]
fn_n = cm[1,0] + cm[2,0]
tp_n = cm[0,0]

precision_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
recall_n = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
specificity_n = tn_n / (tn_n + fp_n) if (tn_n + fp_n) > 0 else 0

print(f"\nClass N (Normal):")
print(f"  True Positives: {tp_n}")
print(f"  False Positives: {fp_n}")
print(f"  True Negatives: {tn_n}")
print(f"  False Negatives: {fn_n}")
print(f"  Precision: {precision_n:.4f}")
print(f"  Recall: {recall_n:.4f}")
print(f"  Specificity: {specificity_n:.4f}")

# For Class P
tn_p = cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]
fp_p = cm[1,0] + cm[1,2]
fn_p = cm[0,1] + cm[2,1]
tp_p = cm[1,1]

precision_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0
recall_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
specificity_p = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0

print(f"\nClass P (Prediabetes):")
print(f"  True Positives: {tp_p}")
print(f"  False Positives: {fp_p}")
print(f"  True Negatives: {tn_p}")
print(f"  False Negatives: {fn_p}")
print(f"  Precision: {precision_p:.4f}")
print(f"  Recall: {recall_p:.4f}")
print(f"  Specificity: {specificity_p:.4f}")

# For Class Y
tn_y = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
fp_y = cm[2,0] + cm[2,1]
fn_y = cm[0,2] + cm[1,2]
tp_y = cm[2,2]

precision_y = tp_y / (tp_y + fp_y) if (tp_y + fp_y) > 0 else 0
recall_y = tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else 0
specificity_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0

print(f"\nClass Y (Diabetes):")
print(f"  True Positives: {tp_y}")
print(f"  False Positives: {fp_y}")
print(f"  True Negatives: {tn_y}")
print(f"  False Negatives: {fn_y}")
print(f"  Precision: {precision_y:.4f}")
print(f"  Recall: {recall_y:.4f}")
print(f"  Specificity: {specificity_y:.4f}")

# Feature Importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance - Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("visualization/decision_tree_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("DECISION TREE IMPLEMENTATION COMPLETE")
print("="*60)