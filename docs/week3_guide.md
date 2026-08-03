# Week 3 Guide: Model Training, Tuning & Evaluation (for absolute beginners)

You already have (from Week 2, in `notebooks/churn_prediction.ipynb`):
- `X_train, y_train` / `X_val, y_val` / `X_test, y_test` — your data split three ways
- `preprocessor` — a not-yet-used object that knows how to scale numbers and one-hot-encode categories

This guide tells you exactly what to type into the empty code cells under sections 6-9 of the notebook (Model Training, Hyperparameter Tuning, Evaluation & Overfitting Check, Model Comparison Table).

## The big idea, in plain English

"Training a model" means: show an algorithm your `X_train` (the customer features) alongside `y_train` (whether they actually churned), and let it find patterns that predict `Churn`. Afterwards you test it on data it's never seen (`X_val`, `X_test`) to see if those patterns actually generalize, or if it just memorized the training data (that's called **overfitting**).

We always **wrap preprocessing + the model together in one `Pipeline`**. Why: it means every time you call `.fit()`, the scaler/encoder is refit correctly on only that call's training data, and you can never accidentally leak information from val/test into training. This is exactly the pattern your Lab9 notebook used with `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(...))])` — we're doing the same thing, just with our `preprocessor` (which handles both scaling AND encoding) instead of a plain `StandardScaler`.

---

## Cell under "6. Model Training"

Type this:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# One Pipeline per model: preprocessing + the model, bundled together
models = {
    "Logistic Regression": Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree":       Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=42))]),
    "Random Forest":       Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))]),
    "SVM":                 Pipeline([("preprocessor", preprocessor), ("classifier", SVC(probability=True, random_state=42))]),
    "MLP (Neural Network)": Pipeline([("preprocessor", preprocessor), ("classifier", MLPClassifier(max_iter=500, random_state=42))]),
}

# Train every model on the training data
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    print(f"{name}: trained")
```

**Why these 5 models?** Logistic Regression is your **baseline** — simplest possible classifier, gives you a number to beat. The other four are the ones the assignment brief explicitly lists as suggested classification algorithms. Having 5 lets you build a genuine comparison table later, which is literally a rubric line item.

**What is `random_state=42`?** It forces the "random" parts of these algorithms (Random Forest picks random subsets of data, MLP starts with random weights) to be reproducible — you'll get the exact same result every time you rerun the cell, instead of slightly different numbers each run. `42` is an arbitrary number, just a convention.

---

## Cell under "7. Hyperparameter Tuning"

**What's a hyperparameter?** A setting you choose *before* training (like how deep a Decision Tree is allowed to grow), as opposed to something the model learns on its own. Picking good hyperparameters can meaningfully improve performance — the brief asks you to tune at least one model with GridSearch or RandomSearch.

Pick your **best-performing model from the cell above** (you'll know once you run evaluation — for now, Random Forest is a safe bet to tune since it usually responds well). Type this:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, 20, None],
}

grid_search = GridSearchCV(
    models["Random Forest"],
    param_grid,
    cv=5,                 # 5-fold cross-validation
    scoring="f1",         # optimize for F1 since classes are imbalanced
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1 score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
```

**Why `classifier__n_estimators` with double underscore?** Because `classifier` is a step *inside* the Pipeline — the double underscore is sklearn's way of saying "reach inside the `classifier` step and set its `n_estimators` parameter."

**Why `cv=5`?** Instead of one train/validation split, it splits your training data 5 different ways and averages the result — a more reliable estimate of how good a hyperparameter combination really is, since it's not relying on luck from one particular split.

**Why `scoring="f1"`?** Because your classes are imbalanced (73.5%/26.5%) — optimizing for plain accuracy would let the model get away with just predicting "No churn" for everyone and still scoring ~73%. F1 punishes that.

---

## Cell under "8. Evaluation & Overfitting Check"

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model, X, y, label):
    preds = model.predict(X)
    print(f"--- {label} ---")
    print("Accuracy :", round(accuracy_score(y, preds), 3))
    print("Precision:", round(precision_score(y, preds), 3))
    print("Recall   :", round(recall_score(y, preds), 3))
    print("F1-score :", round(f1_score(y, preds), 3))
    return preds

# Overfitting check: compare training performance vs validation performance
train_preds = evaluate(best_model, X_train, y_train, "Training set")
val_preds   = evaluate(best_model, X_val, y_val, "Validation set")

# Confusion matrix on validation set — same manual style as Lab9
cm = confusion_matrix(y_val, val_preds)
plt.imshow(cm)
plt.title("Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.show()
```

**How to read the overfitting check:** if Training accuracy/F1 is much higher than Validation (say, 0.95 vs 0.70), the model memorized the training data instead of learning general patterns — that's overfitting. If they're close (e.g. 0.82 vs 0.79), that's healthy. If it IS overfitting, the fix is usually simpler hyperparameters (e.g. lower `max_depth` for trees) — that's why tuning and evaluation are right next to each other.

**Confusion matrix, in plain English:** a 2x2 table. Top-left = correctly predicted "No Churn". Bottom-right = correctly predicted "Churn". The other two corners are your mistakes — top-right is customers you predicted would churn but didn't (false alarm), bottom-left is customers who churned but you missed (the costlier mistake for a real business, since you didn't intervene).

---

## Cell under "9. Model Comparison Table"

```python
comparison = []
for name, pipe in models.items():
    preds = pipe.predict(X_val)
    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1": f1_score(y_val, preds),
    })

comparison_df = pd.DataFrame(comparison).sort_values("F1", ascending=False)
comparison_df
```

This one table does double duty for two rubric lines at once: "Model Selection & Training" (shows you tried multiple algorithms) and "Model Evaluation & Comparison" (shows you compared them properly on multiple metrics, not just one).

---

## Before you run any of this

Run the cells **in order, top to bottom**, every time you open the notebook — later cells depend on variables created in earlier ones (`preprocessor`, `X_train`, etc.). If something errors, it's almost always because a cell above wasn't run first.

## Still waiting on you

Send Lab8 (Trees/Forest) and the other 3 files when you're back — I want to double check the Decision Tree/Random Forest/tuning code above actually matches what your lecturer taught, not just generic sklearn conventions. If there's a mismatch I'll flag exactly what to change.
