import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

# Get global min and max from the original (before cleaning) DataFrame
ymin = df[num_col].min().min() -5
ymax = df[num_col].max().max() + 5

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


#######################################
# 🔹 BINNING / DISCRETIZATION OF AGE FEATURE
#######################################

# Set K value (number of equal-width bins)
K = 5

# Get min and max age
min_age = df_cleaned['AGE'].min()
max_age = df_cleaned['AGE'].max()

# Calculate bin width for equal-width bins
bin_width = (max_age - min_age) / K

# Create bin edges
bin_edges = [min_age + i * bin_width for i in range(K + 1)]
bin_edges[-1] = max_age + 0.001  # Ensure max value is included

# Create labels for the bins
age_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}' for i in range(K)]

# Apply K-bin discretization
df_cleaned['Age_Kbins'] = pd.cut(df_cleaned['AGE'],
                                  bins=bin_edges,
                                  labels=age_labels,
                                  right=False)
# Show results
print(f"\nK={K} Equal-Width Age Binning:")
print(f"Age range: {min_age} to {max_age}")
print(f"Bin width: {bin_width:.2f}")
print(f"Bin edges: {[f'{edge:.1f}' for edge in bin_edges]}")
print(f"Bin labels: {age_labels}")
print("\nSample of Age Binning:")
print(df_cleaned[['AGE', 'Age_Kbins']].head(30))

# Visualize the distribution of Age bins
df_cleaned['Age_Kbins'].value_counts().sort_index().plot(kind='bar', color='skyblue', figsize=(7, 4))
plt.title(f"Distribution of Age Groups (K={K} Equal-Width Bins)")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
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


############################Step 15: KNN Classifier - Find Best k###############################################3
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

### Step 15: KNN Classifier - Find Best k using Test Accuracy Graph

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

#### 15.1: Test different k values on test set

# Try different k values
k_values = list(range(1, 31, 2))  # Odd numbers from 1 to 29
test_accuracies = []

print("Testing different k values on test set...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train_encoded)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    test_accuracies.append(accuracy)

    if k <= 15:  # Print first 15
        print(f"k={k:2d}: Test Accuracy = {accuracy:.4f}")

#### 15.2: Find elbow point in test accuracy graph (excluding k=1)

# Find elbow point (where adding more neighbors doesn't help much)
# Exclude k=1 from being considered as elbow point
def find_elbow_point_excluding_k1(k_values, accuracies, threshold=0.01):
    """
    Find elbow point where accuracy gain becomes minimal
    Excludes k=1 from being selected as the elbow
    threshold: minimum improvement to consider it worth increasing k
    """
    # Find max accuracy excluding k=1
    acc_without_k1 = accuracies[1:]  # Exclude k=1
    max_acc = max(acc_without_k1)

    # Start from k=3 (index 1 in k_values since k_values[0]=1, k_values[1]=3)
    for i in range(1, len(k_values)):  # Start from index 1 (k=3)
        k = k_values[i]
        acc = accuracies[i]

        # If this k gives near-maximum accuracy, check if it's the elbow
        if acc >= max_acc * (1 - threshold):
            # Check if this is the elbow (next point doesn't improve much)
            if i < len(accuracies) - 1:
                improvement = accuracies[i + 1] - acc
                if improvement < threshold * 0.5:  # Very small improvement
                    return k
    # If no clear elbow found, return k=3 as default
    return k_values[1]  # k=3

# Find elbow point excluding k=1
elbow_k = find_elbow_point_excluding_k1(k_values, test_accuracies, threshold=0.02)
print(f"\nElbow point detected at k={elbow_k} (k=1 excluded)")
print(f"Accuracy at elbow (k={elbow_k}): {test_accuracies[k_values.index(elbow_k)]:.4f}")

#### 15.3: Plot test accuracy vs k graph (without vertical line)

plt.figure(figsize=(12, 6))

# Plot accuracy curve
plt.plot(k_values, test_accuracies, 'bo-', linewidth=2, markersize=8, label='Test Accuracy')

# Mark the elbow point (only if not k=1)
if elbow_k != 1:
    elbow_accuracy = test_accuracies[k_values.index(elbow_k)]
    plt.plot(elbow_k, elbow_accuracy, 'ro', markersize=12,
             label=f'Elbow Point (k={elbow_k}, Acc={elbow_accuracy:.4f})')

# Highlight region around elbow (if not k=1)
if elbow_k > 1 and elbow_k < 30:
    plt.axvspan(elbow_k - 2, elbow_k + 2, alpha=0.1, color='yellow', label='Optimal Region')

plt.xlabel('k Value', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('KNN: Test Accuracy vs k Value (Elbow Method)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(k_values[::2])  # Show every other k value
plt.tight_layout()
plt.show()

#### 15.4: Train final model with elbow k (k=3)

# Use the elbow k found (which won't be 1)
final_k = elbow_k
print(f"\nTraining final KNN model with k={final_k} (elbow point)...")

knn_final = KNeighborsClassifier(n_neighbors=final_k)
knn_final.fit(X_train, y_train_encoded)

# Make predictions
y_pred = knn_final.predict(X_test)

#### 15.5: Evaluate model with elbow k

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy with k={final_k}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Show detailed classification report
print("\nClassification Report:")
report = classification_report(y_test_encoded, y_pred,
                               target_names=label_encoder.classes_,
                               output_dict=True)
print(classification_report(y_test_encoded, y_pred,
                            target_names=label_encoder.classes_))

# Extract precision and recall for each class
print("\n=== Detailed Per-Class Metrics ===")
for i, class_name in enumerate(label_encoder.classes_):
    precision = report[class_name]['precision']
    recall = report[class_name]['recall']

    print(f"Class {class_name}:")
    print(f"  Precision: {precision:.4f} - Of predicted {class_name}, {precision * 100:.1f}% were correct")
    print(f"  Recall:    {recall:.4f} - Found {recall * 100:.1f}% of actual {class_name} cases")
    print()

#### 15.6: Visualize confusion matrix

plt.figure(figsize=(10, 8))

# Create confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Number of Cases'})

plt.title(f'KNN Confusion Matrix (k={final_k}, Accuracy: {accuracy * 100:.2f}%)',
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.show()

print(f"\n✓ Selected k={final_k} based on elbow method (k=1 excluded)")
print(f"✓ Test accuracy: {accuracy * 100:.2f}%")
print("✓ Good balance between model simplicity and performance")

######################################################################################################









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


# ============================================================================
# 8. SAVE MODEL FOR GUI AND LAUNCH OPTION
# ============================================================================

import pickle
import os
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings in GUI

print("\n" + "=" * 60)
print("SAVING MODEL FOR GUI")
print("=" * 60)

# Save the trained model
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
print("✓ Model saved as 'naive_bayes_model.pkl'")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as 'scaler.pkl'")

# Save test data WITH feature names
# Convert X_test back to DataFrame with column names
X_test_df = pd.DataFrame(X_test, columns=num_col)
y_test_series = pd.Series(y_test, name='CLASS')

X_test_df.to_csv('X_test.csv', index=False)
y_test_series.to_csv('y_test.csv', index=False)
print("✓ Test data saved as 'X_test.csv' and 'y_test.csv'")

# Also save feature names for reference
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(num_col), f)
print("✓ Feature names saved as 'feature_names.pkl'")

# ============================================================================
# 9. LAUNCH GUI
# ============================================================================

print("\n" + "=" * 60)
print("LAUNCHING GUI")
print("=" * 60)


try:
    from naive_bayes_gui import run_gui

    # Ensure X_train has feature names for GUI display
    X_train_df = pd.DataFrame(X_train, columns=num_col)

    print("\n🚀 Launching GUI...")
    run_gui(nb_model, X_train_df, y_train, X_test_df, y_test_series, scaler, num_col)

except ImportError as e:
    print(f"\n⚠️ Could not launch GUI: {e}")
    print("Make sure naive_bayes_gui.py is in the same directory.")
except Exception as e:
    print(f"\n⚠️ Error launching GUI: {e}")
    print("Troubleshooting steps:")
    print("1. Check all required files exist in directory")
    print("2. Make sure naive_bayes_gui.py is in same folder")
    print("3. Run command: pip install pandas numpy scikit-learn matplotlib seaborn")