import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
import pyfpgrowth


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
df=pd.get_dummies(df,columns=['CLASS'],prefix='Class',dtype=int)


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
df[num_col].boxplot(figsize=(12,6))
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

# Define bins and labels
age_bins = [0, 13, 20, 60, 90]  # Child: 0–12, Teenager: 13–19, Adult: 20–59, Senior: 60–89
labels = ['Child', 'Teenager', 'Adult', 'Senior']

# Apply binning on the original AGE column
df_cleaned['Age_bin'] = pd.cut(df_cleaned['AGE'], bins=age_bins, labels=labels, right=False)

# BMI Binning
bmi_bins = [0, 18.5, 25, 30, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df_cleaned['BMI_bin'] = pd.cut(df_cleaned['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

# More appropriate HbA1c bins for your data range
hba1c_bins = [0, 4.0, 5.7, 6.4, 100]
hba1c_labels = ['Very Low', 'Normal', 'Prediabetes', 'Diabetes']
df_cleaned['HbA1c_bin'] = pd.cut(df_cleaned['HbA1c'], bins=hba1c_bins, labels=hba1c_labels, right=False)

# Show a sample of results
print("\nSample of Age Binning (before outlier detection):")
print(df_cleaned[['AGE', 'Age_bin']].head(30))

# Visualize the distribution of Age bins
df_cleaned['Age_bin'].value_counts().plot(kind='bar', color='skyblue', figsize=(7, 4))
plt.title("Distribution of Age Groups (Child, Teenager, Adult, Senior)")
plt.xlabel("Age Group")
plt.ylabel("Count")
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

#visualizing after removing outliers
df_cleaned.boxplot(figsize=(12,6))
plt.xticks(rotation=45)
plt.show()

#now scaling
scaler=MinMaxScaler()
df_cleaned[num_col] = scaler.fit_transform(df_cleaned[num_col])
df_cleaned[num_col].describe()

df_cleaned[num_col].hist(bins=20, figsize=(15,10))
plt.suptitle("Histogram of Scaled Features")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df_cleaned[num_col].corr(method='spearman'), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Scaled Features")
plt.show()