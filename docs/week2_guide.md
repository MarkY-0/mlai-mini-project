# Week 2 Guide: EDA & Preprocessing (for absolute beginners)

The notebook already has `df = pd.read_csv('../data/Telco-Customer-Churn.csv')` loaded under section 1. Everything below goes into the empty code cells under sections 2-5 (EDA, Cleaning, Transformation & Encoding, Split). This mirrors your Lab5 (Data Cleaning) and Lab6 (Preprocessing) style directly.

---

## Cell under "2. Exploratory Data Analysis"

```python
print("Shape:", df.shape)
df.info()

# Standard NaN check
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))

# Churn class balance
print("\nChurn balance:")
print(df['Churn'].value_counts(normalize=True))

# Visual missing-data check (same style as Lab6 Example 7)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap')
plt.show()
```

**Gotcha to know about:** `df.isnull().sum()` will report **zero** missing values, even though `TotalCharges` actually has 11 blank entries. That's because pandas read them as empty text (`""`), not as a real missing value (`NaN`). Add this line right after to catch it — it's worth explaining in your video, since it shows you actually understand your data rather than trusting `isnull()` blindly:

```python
blank_counts = df.select_dtypes(include='object').apply(lambda col: col.astype(str).str.strip().eq('').sum())
print("\nBlank-string values per column:\n", blank_counts[blank_counts > 0])
```

**Why this matters:** the brief's "Data Preparation" rubric line explicitly rewards "quality of handling missing data" — silently missing 11 broken rows because you trusted one function is exactly the kind of gap a grader looks for.

---

## Cell under "3. Data Cleaning"

```python
# customerID is a unique ID, not a predictive feature - drop it (Lab5 style: keep working copy separate)
clean_df = df.drop(columns=['customerID']).copy()

# Fix TotalCharges: it's text because of the blank entries. Convert to numbers.
clean_df['TotalCharges'] = pd.to_numeric(clean_df['TotalCharges'], errors='coerce')
print("Missing TotalCharges after conversion:", clean_df['TotalCharges'].isnull().sum())

# Median imputation (Lab6 Example 9 style)
clean_df['TotalCharges'].fillna(clean_df['TotalCharges'].median(), inplace=True)

# Check for duplicate rows (Lab6 Section 3)
print("Duplicate rows:", clean_df.duplicated().sum())

# Outlier scan using IQR (Lab6 Example 14 style)
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    outliers = detect_outliers_iqr(clean_df, col)
    print(f"{col}: {len(outliers)} outliers detected")
```

**Judgment call to make yourself (and explain in your video):** should you actually *remove* the outliers the IQR method finds? For this dataset, probably **not** — a customer with very high `TotalCharges` is just a long-tenured, high-paying customer, which is real and meaningful, not an error. Removing them would throw away genuine churn signal. Detecting outliers and then deciding *not* to blindly delete them, with a stated reason, scores better than mechanically dropping rows.

---

## Cell under "4. Feature Transformation & Encoding"

```python
# Encode the target as 0/1
clean_df['Churn'] = clean_df['Churn'].map({'No': 0, 'Yes': 1})

# One-Hot Encoding for categorical columns (Lab6 Example 18 style)
categorical_cols = clean_df.select_dtypes(include='object').columns.tolist()
encoded_df = pd.get_dummies(clean_df, columns=categorical_cols, drop_first=True)

print(f"Columns before encoding: {clean_df.shape[1]}, after: {encoded_df.shape[1]}")
encoded_df.head()
```

**Why `drop_first=True`?** If a column has, say, 3 categories, one-hot encoding normally creates 3 new 0/1 columns. But if you know the values of 2 of them, you already know the 3rd (it's whichever one isn't 1). `drop_first=True` drops that redundant column — same information, fewer columns.

**Note on ordering:** encoding categories here (before splitting) is fine and matches what Lab6 does — it doesn't use any information about the target, so there's no real leakage risk. Scaling is different (see next cell) — that one has to happen *after* the split.

---

## Cell under "5. Train / Validation / Test Split"

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = encoded_df.drop(columns=['Churn'])
y = encoded_df['Churn']

# 60/20/20 split, stratified so all three sets keep the same ~26% churn ratio
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scale numeric columns - fit ONLY on training data (Lab6 Example 21 style, but split-first)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

X_train.head()
```

**Important — this is where we deliberately deviate from Lab6's exact example order.** In Lab6, `StandardScaler` gets `.fit_transform()`'d on the *whole* dataset in one go, because that example was just teaching what scaling does in isolation. For an actual project, that would be **data leakage**: the scaler would learn the mean/spread of your test data too, which means your test performance would look better than it really is. That's why the brief has "Data Leakage Prevention" as its own separately graded line — split first, then `fit_transform` only on `X_train`, then just `.transform()` (no fit) on val/test using the training set's learned mean/spread.

If you can explain *why* that ordering matters in the Q&A, that alone demonstrates real understanding, not just copied syntax.

---

## Checklist before moving to Week 3

- [ ] Run all cells top to bottom, no errors
- [ ] EDA cell shows the blank `TotalCharges` count, not just a generic "0 missing"
- [ ] `clean_df` has no more `customerID`, no more missing values
- [ ] `encoded_df` has more columns than `clean_df` (one-hot expansion)
- [ ] `X_train`, `X_val`, `X_test` shapes roughly 4225 / 1409 / 1409 rows
- [ ] Numeric columns in `X_train` look scaled (small values, mean near 0) when you `.head()` them

Once this runs clean, Week 3 (model training) picks up exactly where `X_train`/`y_train`/`X_val`/`y_val`/`X_test`/`y_test` leave off — no changes needed to the Week 3 guide.
