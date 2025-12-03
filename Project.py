import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
import pyfpgrowth
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
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






from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# 1. NAIVE BAYES MODEL TRAINING
# ============================================================================

# Initialize and train model
nb_model = GaussianNB(priors=None, var_smoothing=1e-9)
print("="*60)
print("TRAINING NAIVE BAYES MODEL")
print("="*60)
nb_model.fit(X_train, y_train)

# Make predictions
y_test_pred = nb_model.predict(X_test)

# ============================================================================
# 2. BASIC METRICS
# ============================================================================

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"\nOverall Model Metrics:")
print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# ============================================================================
# 3. METRICS VISUALIZATION FOR WHOLE MODEL
# ============================================================================

plt.figure(figsize=(6, 4))
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]
colors = ['green', 'blue', 'orange']

bars = plt.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
plt.ylabel('Score')
plt.title('Naive Bayes: Overall Model Metrics', fontsize=12, fontweight='bold')
plt.ylim([0, 1.1])

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom')

plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("visualization/NaiveBayes_Overall_Metrics.png", dpi=300)
plt.show()

# ============================================================================
# 4. CONFUSION MATRIX
# ============================================================================

classes = ['N', 'P', 'Y']
cm = confusion_matrix(y_test, y_test_pred, labels=classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Naive Bayes - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("visualization/NaiveBayes_Confusion_Matrix.png", dpi=300)
plt.show()

# ============================================================================
# 5. PER-CLASS ACCURACY
# ============================================================================

print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

# Calculate from confusion matrix
class_acc = np.diag(cm) / cm.sum(axis=1)

for i, cls in enumerate(classes):
    print(f"Class {cls}: {class_acc[i]:.3f} ({class_acc[i]*100:.1f}%)")

# Simple bar chart
plt.figure(figsize=(5, 4))
plt.bar(classes, class_acc, color=['blue', 'orange', 'red'], alpha=0.7)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Naive Bayes: Accuracy per Class')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("visualization/NaiveBayes_PerClass_Accuracy.png", dpi=300)
plt.show()

# ============================================================================
# 6. MISCLASSIFIED SAMPLES
# ============================================================================

print("\n" + "="*60)
print("MISCLASSIFICATION SUMMARY")
print("="*60)

misclassified = sum(y_test != y_test_pred)
error_rate = misclassified / len(y_test)
print(f"Misclassified samples: {misclassified}/{len(y_test)}")
print(f"Error rate: {error_rate:.3f} ({error_rate*100:.1f}%)")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("NAIVE BAYES - FINAL SUMMARY")
print("="*60)
print(f"Model: Gaussian Naive Bayes")
print(f"Overall Accuracy: {accuracy:.3f}")
print(f"Best predicted class: {classes[np.argmax(class_acc)]} ({np.max(class_acc):.3f})")
print(f"Worst predicted class: {classes[np.argmin(class_acc)]} ({np.min(class_acc):.3f})")
print(f"Error rate: {error_rate:.3f}")

print("\n" + "="*60)
print("VISUALIZATIONS SAVED:")
print("="*60)
print("1. NaiveBayes_Overall_Metrics.png - Accuracy, Precision, Recall")
print("2. NaiveBayes_Confusion_Matrix.png - Confusion matrix")
print("3. NaiveBayes_PerClass_Accuracy.png - Accuracy per class")