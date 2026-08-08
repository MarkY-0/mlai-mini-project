# AI Usage Log

This log records where Generative AI tools were used during the mini project, what was
adopted or rejected, and how the output was checked. It supports the "Responsible Use of
GenAI" section of the assessment and the Declaration of Academic Integrity in the appendix.

Tool used: Claude (Anthropic)

---

## 1. Dataset selection and verification

**Prompt:** "Recommend a project scenario and dataset for a churn prediction mini project,
and confirm the dataset is real and not AI generated."

**AI output:** Suggested Customer Churn Prediction using the IBM Telco Customer Churn dataset
(7,043 rows, 21 columns) and gave the Kaggle / IBM source URL.

**Adopted / Rejected:** Adopted.

**How we validated it:** We did not take the AI's word that the dataset was suitable. We
checked the file ourselves in pandas: 7,043 rows confirmed, 0 duplicate customer IDs, only 11
blank `TotalCharges` values, and a 73.5% / 26.5% class balance. We also confirmed the source
URL loads the same file, since the brief requires a real dataset with a stated source.

---

## 2. Concepts used that go beyond the lab material

These were suggested by AI and kept because they genuinely improved the project. They are
recorded here because they were not taught in Labs 5 to 11.

### a) `class_weight="balanced"` in Logistic Regression

**Prompt:** "Our churn model has low recall. Only 26% of customers churn. What can we change?"

**AI output:** Explained that with an imbalanced target the model leans towards predicting the
majority class, and suggested `class_weight="balanced"` so the minority class is weighted more
heavily during training.

**Adopted / Rejected:** Adopted.

**Why and how we checked it:** We did not just switch it on. We put it into the GridSearchCV
parameter grid as `[None, "balanced"]` and let cross-validation decide. `"balanced"` won on F1,
so the choice is backed by our own CV result rather than by the AI's recommendation.
Lab 10 covered regularization (Ridge and Lasso) but not class weighting, so this is new
material for us.

### b) Choosing the decision threshold instead of leaving it at 0.5

**Prompt:** "Recall is still low. Is 0.5 always the right cut-off for classification?"

**AI output:** Explained that 0.5 is only a default, and that the threshold can be moved to
trade precision against recall depending on which mistake is more costly.

**Adopted / Rejected:** Adopted, but implemented using the method from Lab 9 Part C.

**Why and how we checked it:** Lab 9 already taught us to loop over thresholds and plot
precision and recall, so we extended that same loop to also record F1 and pick the best value
(0.60). The AI initially suggested using `precision_recall_curve` from sklearn, which we
**rejected** because it was not covered in class and we could get the same answer with the
loop we already understood. Moving the threshold raised test recall from about 0.66 to 0.714.

### c) Three-way train / validation / test split

**Prompt:** "The brief asks for training, validation and testing sets but our labs only did a
train/test split. How do we do three?"

**AI output:** Suggested calling `train_test_split` twice, first to take out 40%, then
splitting that 40% in half.

**Adopted / Rejected:** Adopted, because the project brief explicitly requires three sets.

**How we checked it:** We printed the shape and the churn rate of all three sets to confirm the
60/20/20 proportions and that `stratify` kept the churn rate at 0.265 in each one.

### d) Cross-validation to confirm the model choice

**Prompt:** "How do we know Logistic Regression really is the best model and not just lucky
with our particular split?"

**AI output:** Suggested `cross_val_score` with 5 folds to check the ranking is stable.

**Adopted / Rejected:** Adopted.

**How we checked it:** We already knew `cv=5` from GridSearchCV in Lab 11, so this was a small
step from familiar material. The result supported the earlier choice: Logistic Regression was
first in all 5 folds with the smallest standard deviation (0.004).

---

## 3. AI suggestions we rejected

- **`ColumnTransformer` with `OneHotEncoder`.** The AI first wrote the preprocessing using
  `ColumnTransformer`. We replaced it with `pd.get_dummies` plus a `Pipeline` containing
  `StandardScaler`, because that is the combination taught in Lab 6 and Lab 11 and we can
  explain it properly in the Q&A. The Pipeline still fits the scaler on training data only, so
  data leakage is still prevented.
- **`ConfusionMatrixDisplay`.** Replaced with `plt.imshow` plus a text loop, which is the
  method from Lab 9 and Lab 11.
- **`RandomizedSearchCV`.** Replaced with `GridSearchCV` on a smaller grid, since only
  GridSearchCV was taught.
- **XGBoost / LightGBM.** The AI mentioned these could score slightly higher. We rejected them
  because the brief lists Logistic Regression, Decision Trees, Random Forest, SVM and Neural
  Networks as the expected algorithms, and we would not be able to justify a library we have
  not studied.

---

## 4. An AI error we caught

While reviewing the feature importance output, the AI stated that the engineered
`AvgChargePerMonth` feature had "broken" the model because the `MonthlyCharges` coefficient
came out negative, which looks wrong.

We tested this ourselves by re-running the model with and without the engineered features. The
`MonthlyCharges` coefficient was negative in **both** versions (-1.50 without the new features,
-0.89 with them), so the engineered feature was not the cause. The real reason is that
`MonthlyCharges` overlaps with the internet and service columns, since the monthly bill is
largely decided by which services the customer has.

We also measured whether the engineered features helped at all rather than assuming it: test
recall went from 0.658 without them to 0.714 with them, so we kept them. This is noted as a
limitation in Part 15 of the notebook so we do not over-claim from the charge coefficients.

**Lesson:** the AI's explanation sounded confident and technical but was wrong. Checking it
took one experiment and changed our conclusion.

---

## 4b. A second AI error we caught: the lifespan model was trained on the wrong customers

When we added the regression model (Part 16) to estimate customer lifespan, the AI suggested
training it on the customers who had already churned, reasoning that only they have a known final
lifespan. That sounds correct and we accepted it at first.

Testing the app on a deliberately high-risk customer showed it was wrong. A customer with an 87%
churn probability paying $105 a month came out with a revenue at risk of exactly **$0.00**. The
model had learned "how quickly do people leave" rather than "how long would this person stay", so
it predicted a lifespan shorter than the time the customer had already been with the company, and
the remaining months collapsed to zero. The customers the company should care about most were
being reported as worth nothing.

We re-trained the model on the retained customers instead and compared the two properly:

| Trained on | R2 | Revenue at risk for the test customer |
|---|---|---|
| Churned customers | 0.584 | $0.00 |
| Retained customers | 0.622 | $1,509.42 |

The retained version is both conceptually right (the money lost is the money they *would* have
paid) and measurably better. We also found the Linear Regression was predicting impossible
lifespans (-0.4 to 81.8 months when the data only spans 0 to 72), so we clip the output and note
that as a limitation rather than hiding it.

**Lesson:** the bug was invisible in the metrics. R2 looked fine and the notebook ran without
errors. It only showed up because we tested the app with a realistic worst-case customer instead
of the default values.

---

## 4c. Catching our own inconsistency: recall vs the metric we were optimising

While reviewing the app we asked why the threshold was 0.60 and whether it could just be hard
coded at 0.5. Working through it exposed a contradiction we had written ourselves: the notebook
claimed recall mattered more than precision, but then selected the threshold by maximising **F1**,
which weights precision and recall equally. The two statements cannot both be true.

We fixed it by switching the selection criterion to the **F2-score**, which weights recall twice
as heavily. The threshold moved from 0.60 to 0.30 and the results changed a lot:

| | F1 (threshold 0.60) | F2 (threshold 0.30) |
|---|---|---|
| Churners caught | 267 of 374 | 347 of 374 |
| Churners missed | 107 | 27 |
| Recall | 0.714 | 0.928 |
| Precision | 0.546 | 0.431 |
| Accuracy | 0.767 | 0.655 |

We then went one step further and checked whether "recall matters more" was actually true in money
terms rather than just asserted (Part 19). Using the customer values from Part 17, the best
threshold turned out to depend entirely on what the campaign costs: 0.30 for a $20 automated
email, 0.55 for a $50 discount, 0.75 for a $100 phone call, and a loss of about $10,900 at our
0.30 if offers only work one time in ten.

**Lesson:** the AI wrote a justification ("recall matters more, so we tuned the threshold") that
read convincingly and did not match the code underneath it, and neither of us noticed until we
questioned a number in the app. It also taught us that the accuracy drop from 0.767 to 0.655 is
not a failure, but it does need the cost argument to defend it, otherwise it just looks like a
worse model.

---

## 4d. Questioning the false alarm count led to the biggest improvement

Looking at the confusion matrix we noticed the model raised 459 false alarms, meaning 44% of loyal
customers would receive a retention offer. That felt wrong, so we asked whether it really was.

Two things came out of checking it.

First, our cost model had been too generous. We had treated a false alarm as a flat $20 to $100
admin cost, but if the retention offer is a discount then giving it to somebody who was staying
anyway is lost revenue on a full paying customer, roughly $155 on average for 20% off for a year.
Re-running Part 19 with that cost showed our threshold of 0.30 **loses about $50,900**, and every
threshold below 0.70 loses money.

Second, and more useful, the problem was not the threshold value but the idea of using a single
threshold at all. A threshold only looks at how likely somebody is to leave and ignores how much
they are worth, so it treats a $20 a month customer the same as a $110 one. Deciding customer by
customer instead (Part 20) gave:

| Strategy | Contacted | Caught | False alarms | Profit |
|---|---|---|---|---|
| Threshold 0.30 | 806 | 347 | 459 | $47,985 |
| Contact only if it pays off | 450 | 231 | 219 | **$65,775** |

Half the false alarms, 356 fewer people contacted, and 37% more profit. With the discount costing
it is the only strategy that makes money at all.

**Lesson:** this was not an AI suggestion that we checked, it came from questioning a number that
looked wrong in our own results. The AI had built the threshold approach and written a confident
justification for it, and neither the notebook nor the AI flagged that we had computed each
customer's value in Part 17 and then ignored it when deciding who to contact.

---

## 4e. A number we could not reproduce, caught while writing the slides

While building the presentation a team member tried to verify the claim that our engineered
features improved test recall from 0.658 to 0.714. He re-ran it several ways and could not
reproduce it anywhere, so he removed the figure from the slide rather than present a number he
could not stand behind.

He was right twice over. First, the comparison had never been in the notebook at all: Part 5
created the features and plotted churn rate against them, but nothing measured whether the model
was actually better with them. Second, and worse, the original figure was **confounded**. The two
sides of the comparison had each picked their own decision threshold (0.65 without the features,
0.60 with them), so the recall gap was partly the threshold moving, not the features doing
anything.

We then ran it properly, holding the threshold fixed at 0.30 for both sides and changing nothing
but the features. That comparison is now in Part 5:

| Feature set | Recall | Precision | F1 | ROC AUC |
|---|---|---|---|---|
| Original only | 0.9198 | 0.4185 | 0.5753 | 0.8319 |
| With engineered | 0.9278 | 0.4305 | 0.5881 | 0.8376 |

The features do help, on every measure including ROC AUC which does not depend on the threshold,
but by well under one percentage point of recall rather than the 5.6 points originally claimed.

**Lesson:** this is the error we are most pleased to have caught, because it was ours to catch.
The number came from AI-assisted analysis, it sounded reasonable, and it had already been copied
into our notes and our draft slides before anybody tried to reproduce it. Nothing flagged it. It
only surfaced because a team member refused to present a figure he could not find in the code.

---

## 5. Where AI was used for code and writing

- Debugging a `NameError` caused by running notebook cells out of order after a kernel restart.
  Fix was to make each cell import what it uses.
- Explaining what precision, recall, F1 and the confusion matrix actually mean, which we then
  wrote up in our own words in the notebook and slides.
- Drafting wording for the conclusion section. All numbers quoted in it were taken from our own
  executed output, not from the AI, and we corrected the draft where it listed factors (Tech
  Support and Online Security) that turned out to be much weaker in our final model than the
  draft claimed.

---

## 6. Reflection

The most useful thing AI gave us was direction: it pointed out that our recall was the real
weakness when we were only looking at accuracy. The least reliable thing was its explanations
of *why* a result happened, which sounded convincing but was wrong at least once (section 4).

Our rule for the rest of the project was that AI could suggest what to try, but the notebook
only keeps a change if our own executed output shows it helped. Every score, coefficient and
threshold reported in the notebook comes from running the code ourselves.
