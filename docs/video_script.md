# Video Presentation Script — Customer Churn Prediction

4 sections, ~5 minutes each = ~20 minutes total. Every figure below is taken directly from the executed notebook (`churn_prediction.ipynb`) or from a live, verified run of `app.py` — nothing is estimated or invented.

---

## Section 1 — Data Exploration & GenAI Support (Notebook Parts 1-7)

You can't trust a model if you can't trust the data going into it, and ours had a couple of problems hiding in plain sight. I'll walk you through how we found them.

We're predicting customer churn for a telecom company — figuring out ahead of time which customers are about to cancel, so the company can step in with a retention offer before they leave, not after.

The dataset is the IBM Telco Customer Churn dataset, a real dataset, 7,043 customers, 21 columns, source linked on our slide and in the notebook header. We asked our AI assistant to help us pick a scenario and dataset for this kind of project, and it suggested this one. We didn't take that on faith, though. We loaded the file ourselves and checked it: 7,043 rows, zero duplicate customer IDs, and the source link actually pointed to that file.

[show the churn class balance chart] First thing we checked was the class balance. 73.5% of customers stayed, 26.5% churned. That number matters more than it looks. It's the reason we later used class weighting during training, and the reason we judged our models on F2 instead of accuracy. Everything downstream traces back to this one number.

Then we went looking for problems, and found one hiding in plain sight. `df.isnull()` said there were zero missing values in the whole dataset. That was wrong. `TotalCharges` is a text column, and 11 rows had a blank space instead of a number, so pandas never flagged them as missing at all, we had to check for blank strings separately to catch it. Once we converted the column to numbers, those 11 became real gaps, and we filled them with the median, since it isn't skewed by a handful of extreme values the way a mean would be. It's not a coincidence either: every one of those 11 rows belongs to a brand-new customer with zero months of tenure. They simply haven't been billed yet.

We also checked for outliers with the IQR method on tenure, monthly charges, and total charges, and found zero in all three. Makes sense once you think about it, tenure caps at 72 months and the charges are ordinary monthly bills, so there's no realistic way for an entry error to land outside those bounds.

Beyond cleaning, we built three new features instead of just feeding in the raw columns: how many optional services a customer has signed up for, their average monthly spend, and a tenure bucket, new, medium, or long-term. We didn't add these because it felt like the thing to do, we measured it. In an isolated test, adding these three features moved recall from 0.658 to 0.714. That number shifts again later once we correct our decision threshold, but this was the test that told us the features were worth keeping.

For the 16 categorical columns, we one-hot encoded with `drop_first=True`. Keep every dummy column and one becomes perfectly predictable from all the others, so we drop one per category to avoid that redundancy.

Last part on my end is the split. The brief asks for three sets, training, validation, and test, but our labs only ever covered a two-way split. That specific idea came from asking our AI assistant how to extend it, and it suggested splitting twice: 60% off for training, then cutting the remaining 40% in half. We didn't just trust that it worked structurally, we printed the shape and the churn rate of all three sets afterward to confirm the 60/20/20 split held and the 26.5% churn rate stayed identical in each one. That mattered, because the validation set is where every model and threshold gets compared, and the test set only gets opened once, right at the end, so its score stays honest.

That's the foundation everything else sits on. Over to [Name] for how we actually trained the models on top of it.

---

## Section 2 — Model Training & AI-Assisted Development (Notebook Parts 8-10)

Five models went into this competition. Only one came out the winner, and it wasn't the fancy one.

We trained five: Logistic Regression, Decision Tree, Random Forest, SVM, and a Neural Network, an MLP. Not an arbitrary list, it's the exact set of classification algorithms the module brief points to, plus Random Forest, a natural extension of the tree-based work we'd already covered.

Every one of these sits inside a scikit-learn Pipeline, StandardScaler first, model second. We did it this way rather than scaling the whole dataset upfront because of data leakage: if you fit the scaler on everything before splitting, the mean and spread of your validation and test data leak into what the model trains on, and your evaluation stops being honest. Inside a Pipeline, the scaler only ever sees the training fold it's currently working with. That holds true even inside GridSearchCV, which refits per fold automatically. On the loss side, Logistic Regression and the others are minimizing cross-entropy loss, the standard loss for a model predicting a churn probability.

Before any tuning, we ran all five once to get a baseline. Logistic Regression already led with an F1 of about 0.617, completely untouched. Worth remembering, the simplest model on our list was already winning before we did anything to help it.

[show the GridSearchCV results table] Then we tuned three of them, Logistic Regression, Random Forest, and Decision Tree, with GridSearchCV, five-fold cross-validation, scored on F1 rather than accuracy, because with a 73/26 class split, accuracy lets a model get away with predicting "no churn" for everyone. For Logistic Regression specifically, we searched over `class_weight`, either left alone or set to "balanced," because with this kind of imbalance a model naturally leans toward the majority class. Our labs hadn't covered that particular fix, they'd only gone as far as L1 and L2 regularization, so this came from asking our AI assistant what else was worth trying. We didn't switch it on just because it suggested it, we put it in the grid as an option and let cross-validation decide. "Balanced" won, with C equal to 10.

The Decision Tree result is the clearest example of regularization actually doing something visible. Left completely untuned, it hit 0.997 F1 on training data and dropped to 0.495 on validation, a tree that memorized every training row instead of learning a pattern. Once GridSearchCV picked `max_depth=6` and `min_samples_leaf=20`, that gap fell to 0.035. That's regularization, concretely, not just a term we're quoting.

We didn't tune SVM or the neural network the same way, and I want to be upfront about that rather than skip past it. When we checked the overfitting gap across all five afterward, the MLP had the worst one by a distance, 0.268, against 0.035 for the tuned tree. That gap is exactly why it didn't get chosen, so rather than spending more time tuning a model that was already clearly overfitting, we used that result as evidence against it and moved on.

One more thing worth saying. Our first version of the preprocessing used `ColumnTransformer`. We rewrote it using `pd.get_dummies` with a Pipeline instead, because that combination is what our labs actually taught, and it means we can explain every line of it if asked. Same story with `RandomizedSearchCV`, we stuck with GridSearchCV since it's the method covered in class.

That's the training side. Over to [Name] for how we decided which of these five actually won.

---

## Section 3 — Performance Analysis & Validation (Notebook Parts 11-15)

Anyone can pick a model that looks good once. My job was proving Logistic Regression wasn't just lucky.

[show the model comparison chart] Once all five were tuned, we compared them on the validation set using accuracy, precision, recall, and F1 together, never just one number, because a single metric hides a lot. Logistic Regression came out on top with an F1 of 0.633, ahead of Decision Tree at 0.579, the neural network at 0.573, Random Forest at 0.563, and SVM at 0.558.

Because that's one particular split, we didn't stop there. We ran five-fold cross-validation across the training and validation data combined, and Logistic Regression won every single fold, mean F1 of 0.630, and the lowest standard deviation of the five, at 0.004. The other four bounced around far more between folds. This isn't a result that happened to favor Logistic Regression once, it's consistently the best model regardless of which slice of data it sees.

Then came the threshold. By default, a model calls it "churn" above 50% probability, but that's a convention, not a rule. We swept thresholds from 0.10 to 0.95 and originally picked the one that maximized F1, which landed at 0.60. Going back through our own reasoning afterward, we caught a contradiction we'd written ourselves: we'd been arguing all along that missing a churner is worse than a false alarm, but F1 treats precision and recall as equally important, so picking a threshold that way didn't actually match our own argument. We switched to F2, which weights recall twice as heavily, and the threshold moved to 0.30. Recall jumped from 0.714 to 0.928, we catch far more churners, at the cost of accuracy dropping from 0.767 to 0.655. [Name] will get into why that trade is worth it in the next section.

With the model and threshold locked in from validation, we opened the test set exactly once. Final numbers: accuracy 0.655, precision 0.431, recall 0.928, F1 0.588, F2 0.754. Out of 374 real churners in the test set, we catch 347 and miss 27, at the cost of 459 false alarms.

I want to address that accuracy number directly, because on its own it looks bad, 0.655 is actually lower than the 73.5% you'd get by predicting "no churn" for every customer. We're not hiding from that. We deliberately traded accuracy for recall, because missing a churner costs the company real revenue, and a false alarm just costs an unnecessary discount. Accuracy alone doesn't capture that trade-off, which is exactly why the brief asks for multiple metrics instead of one.

[show the feature importance chart] Last part of my section is what actually drives the prediction. Logistic Regression gives us a coefficient per feature, and the biggest ones were fiber optic internet, pushing customers toward churning, and tenure, keeping them. One number looked wrong at first though: `MonthlyCharges` came out negative, meaning higher bills were linked to lower churn, which doesn't match common sense. Our first theory, from our AI assistant, was that our engineered `AvgChargePerMonth` feature had broken something. So we tested that directly, retrained the model with and without the engineered features, and the `MonthlyCharges` coefficient was negative in both versions. The real explanation is multicollinearity: the monthly bill is largely determined by which internet type and add-ons a customer has, so once fiber optic is already in the model with its own strong coefficient, `MonthlyCharges` shifts to balance it out. That first explanation sounded confident and technical, and it was wrong. Catching it took one experiment.

Over to [Name] to turn all of this into something the business can actually act on.

---

## Section 4 — Business Insights & Emerging Technologies (Notebook Parts 16-21 + Conclusion)

Knowing a customer might leave is only half the problem. The real question is what we should actually do about it, and what it costs to get that wrong.

A churn probability alone doesn't say how much a customer is worth. So we built a second model, a Linear Regression, estimating how many more months a customer is likely to stay. This one's minimizing mean squared error instead of cross-entropy, since we're predicting a number now, not a category. We multiplied that prediction by monthly charges to get revenue at risk per customer. Across the test set, that's about $390,000 total, $277 per customer on average.

Getting that regression model right took a genuine mistake first. Our first instinct, which our AI assistant agreed with, was to train it on customers who had already churned, since they're the only ones with a known, final lifespan. Sounds reasonable. It was wrong. We tested it on a deliberately high-risk customer, 87% churn probability, paying $105 a month, and their revenue at risk came back as exactly zero dollars. The model had learned how quickly people leave, not how long someone would realistically stay, so for our riskiest customers it predicted a lifespan shorter than the time they'd already been with the company. The metrics didn't catch this, R-squared looked perfectly reasonable. We only found it by testing the app with a realistic worst-case customer instead of trusting the score. Retraining on retained customers fixed it and scored better too, R-squared of 0.622 against 0.584.

With a dollar value per customer, we could rank properly. If the retention team can only call 100 people, ranking by churn probability alone protects about $141,559 of revenue. Ranking by revenue at risk instead protects $178,794, 26% more, for the same 100 phone calls. And the right decision threshold turned out to depend entirely on what the offer costs: 0.30 is close to optimal for a cheap $20 email, but for a $100 phone call it's closer to 0.75, which is why the app exposes it as a slider instead of a fixed number.

That led to the most useful finding in the project. Our threshold was flagging 459 false alarms out of 1,409 test customers, 44% of loyal customers getting an unnecessary offer. A threshold only looks at how likely someone is to leave, it ignores what they're worth, so it treats a $20-a-month customer identically to a $110-a-month one. Deciding per customer instead, only contact them if their value times their risk beats what the offer costs, contacts 450 people instead of 806 on a flat $50 offer, roughly halves the false alarms, and makes 37% more profit. With a realistic 20% discount, the threshold approach actually loses money, while targeting still turns a profit, because it automatically skips customers whose bill is too small to justify the discount.

Let me show you this running live rather than just describing it. [switch to the app] Take a brand new customer, four months in, fiber optic, month-to-month, paying by electronic check, $92.88 a month, close to our riskiest profile on paper. The app puts them at 63.9% likely to churn, $742.94 in revenue at risk. Cost to contact them: fifty dollars. Expected gain: $222.88. So yes, worth contacting, and the app goes one step further, it recommends moving them onto a two-year contract specifically, which for this customer drops their risk from 63.9% down to 26.7%. That's the full pipeline, raw customer details in, a specific costed recommendation out, running live, not just sitting in a notebook. That's really the emerging-technology angle here: less static reporting, more tools a non-technical team could actually open and use.

Four recommendations came out of this. Move month-to-month customers onto longer contracts, it's the strongest lever the company controls. Investigate why fiber optic customers churn more, most likely price or reliability, worth a survey. Focus retention effort in the first year, that's where the risk is highest. And nudge electronic check users onto automatic payment, a smaller effect, but cheap to act on.

Two honest limits before we close. The 30% save rate behind every profit figure here is an assumption, not something measured, if the real number is lower, the campaign math changes. And the what-if tool shows correlation, not proof, moving someone onto a longer contract doesn't guarantee they'll stay, it just means people like them tend to.

That's everything from us, thanks for watching.

---

## Notes on this revision

- **Live demo added to Section 4**, using a real customer profile run through the actual app just now, not invented numbers: new customer (4 months tenure), fiber optic, month-to-month, paperless billing, electronic check, $92.88/month → 63.9% churn probability, $742.94 revenue at risk, worth contacting (+$172.88 net), recommended offer moves risk from 63.9% to 26.7%. **Is it needed?** Yes — the brief's "Extra Wow Factor" line literally asks for an interactive application; showing it running live turns a claim into proof, and costs nothing extra since the app already works.
- **MSE added** for the regression model's loss function (Section 4) — the brief asks you to name the loss for both classification and regression; only cross-entropy was covered before.
- **Found and ruled out a false alarm**: initially suspected a stale-data bug in `app.py`'s Revenue-at-risk tab. Investigated the source (logic was correct) and re-tested cleanly — the app is fully consistent. That earlier anomaly was my browser automation catching a mid-rerun state, not your code. Confirmed, not flagging it as an issue.
- **Deliberately left out**: the AI log's routine NameError-debugging note. It's real but minor compared to the four stronger "caught AI being wrong" stories already in the script, and including every AI log entry would dilute those and blow the time budget. Say if you want it added back somewhere.
- Openers rewritten to lead with a hook (a tension, a stat, a claim) instead of "hi, I'm going to talk about X."
- Removed a few spots where AI mentions were feeling listy/tacked-on; folded them into the sentence they belong to instead.
