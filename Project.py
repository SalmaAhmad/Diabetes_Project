import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
import pyfpgrowth

#DEFINITIONS THRESHOLDS THAT CAN VE TWEAKED LATER
MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.6


df=pd.read_csv("Diabetes_Project\Dataset_Diabetes.csv")

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

df_cleaned = df[~outlier_mask.any(axis=1)]

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


#Build transactions for association rule mining
df_assoc = df_cleaned.copy()

#discretize numerical columns
disc_cols = []
for col in num_col:
    try:
        df_assoc[col + '_BIN'] = pd.qcut(df_assoc[col], q=3, labels=["LOW", "MEDIUM", "HIGH"])
        disc_cols.append(col + '_BIN')
    except ValueError:
        pass

#collect all categorical columns
cat_items_cols = []
if "Gender" in df_assoc.columns:
    cat_items_cols.append("Gender")
# pick any Class_* columns if they exist
class_onehots = [c for c in df_assoc.columns if c.startswith("Class_")]
cat_items_cols.extend(class_onehots)

#create transactions
transactions = []
for index, row in df_assoc.iterrows():
    items = []
    # discretized numerics
    for col in disc_cols:
        if pd.notna(row[col]):
            base = col.replace('_BIN', '')
            items.append(f"{base}_{row[col]}")
    #gender
    if "Gender" in row.index and pd.notna(row["Gender"]):
        items.append(f"Gender={row['Gender']}")
    #class onehots
    for col in class_onehots:
        if row.get(col, 0) == 1:
            items.append(f"CLASS={col.split('_',1)[1]}")
    transactions.append(items)

n_tx = len(transactions)
print(f"\n[Assoc] Built {n_tx} transactions with {len(disc_cols)} discretized features.")

#Apply Apriori Algorithm
apriori_rules = list(apriori(transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE))
print(f"[Apriori] Found {len(apriori_rules)} rules (min_sup={MIN_SUPPORT}, min_conf={MIN_CONFIDENCE})")


# Convert apyori output → DataFrame ( support, confidence, lift)
ap_rows = []
for r in apriori_rules:
    supp = r.support
    for os in r.ordered_statistics:
        if len(os.items_base) == 0:  # skip empty antecedent
            continue
        ap_rows.append({
            "antecedent": tuple(sorted(os.items_base)),
            "consequent": tuple(sorted(os.items_add)),
            "support": supp,
            "confidence": os.confidence,
            "lift": os.lift
        })

df_rules_ap = pd.DataFrame(ap_rows).sort_values(["lift","confidence","support"], ascending=False)
print("\n=== Top Apriori rules ===")
print(df_rules_ap.head(20).to_string(index=False))