import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

model = joblib.load("models/churn_pipeline.joblib")
lifespan_model = joblib.load("models/lifespan_model.joblib")
model_info = joblib.load("models/churn_metadata.joblib")

feature_columns = model_info["feature_columns"]
regression_columns = model_info["regression_columns"]
category_options = model_info["category_options"]
numeric_ranges = model_info["numeric_ranges"]
service_cols = model_info["service_cols"]
best_model_name = model_info["best_model_name"]
best_reg_name = model_info["best_reg_name"]
test_metrics = model_info["test_metrics"]
regression_metrics = model_info["regression_metrics"]
default_threshold = round(model_info["threshold"], 2)
save_rate = model_info["save_rate"]
flat_offer_cost = model_info["flat_offer_cost"]

st.title("Customer Churn Predictor")
st.write("ET0737 Mini Project. Classifier:", best_model_name, "| Lifespan model:", best_reg_name)

st.sidebar.header("Classification (test set)")
st.sidebar.write("Accuracy :", round(test_metrics["Accuracy"], 3))
st.sidebar.write("Precision:", round(test_metrics["Precision"], 3))
st.sidebar.write("Recall   :", round(test_metrics["Recall"], 3))
st.sidebar.write("F1-score :", round(test_metrics["F1-score"], 3))
st.sidebar.write("F2-score :", round(test_metrics["F2-score"], 3))
st.sidebar.caption("Recall is high on purpose. Missing a churner costs far more than "
                   "sending a discount to someone who was staying anyway.")

st.sidebar.header("Regression (test set)")
st.sidebar.write("MAE :", round(regression_metrics["MAE"], 2), "months")
st.sidebar.write("RMSE:", round(regression_metrics["RMSE"], 2), "months")
st.sidebar.write("R2  :", round(regression_metrics["R2"], 3))


def encode_customer(customer):
    """Turn one customer dictionary into the same columns used during training."""
    row = pd.DataFrame([customer])

    row["TotalServices"] = (row[service_cols] == "Yes").sum(axis=1)
    row["AvgChargePerMonth"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["TenureGroup"] = pd.cut(
        row["tenure"], bins=[-1, 12, 36, 72], labels=["New", "Medium", "Long"]
    )

    categorical_cols = row.select_dtypes(exclude=["number"]).columns.tolist()
    row = pd.get_dummies(row, columns=categorical_cols, drop_first=False)

    # Columns this customer does not have are filled with 0, which is how the
    # reference categories dropped by drop_first=True are represented.
    return row.reindex(columns=feature_columns, fill_value=0)


def set_category(row, prefix, chosen_column=None):
    changed = row.copy()
    for col in changed.columns:
        if col.startswith(prefix + "_"):
            changed[col] = 0
    if chosen_column is not None:
        changed[chosen_column] = 1
    return changed


def add_service(row, service_column):
    changed = row.copy()
    if changed[service_column].iloc[0] == 0:
        changed[service_column] = 1
        changed["TotalServices"] = changed["TotalServices"] + 1
    return changed


st.header("Customer details")

col1, col2, col3 = st.columns(3)

# st.radio is used instead of st.selectbox because a Streamlit selectbox is a
# searchable box that the user can type into. Radio buttons only allow the options
# we give them, so nobody can enter a value the model has never seen.

with col1:
    st.write("**Demographics**")
    gender = st.radio("Gender", category_options["gender"], horizontal=True)
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Has partner", category_options["Partner"], horizontal=True)
    dependents = st.radio("Has dependents", category_options["Dependents"], horizontal=True)
    tenure = st.slider(
        "Tenure (months)",
        int(numeric_ranges["tenure"]["min"]),
        int(numeric_ranges["tenure"]["max"]),
        int(numeric_ranges["tenure"]["median"])
    )

with col2:
    st.write("**Services**")
    phone_service = st.radio("Phone service", category_options["PhoneService"], horizontal=True)

    # If the customer has no phone line, "Multiple lines" can only be
    # "No phone service", so we set it instead of letting the user pick something
    # that could never appear in the real data.
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.caption("Multiple lines: No phone service (set automatically)")
    else:
        multiple_lines = st.radio("Multiple lines", ["No", "Yes"], horizontal=True)

    internet_service = st.radio(
        "Internet service", category_options["InternetService"], horizontal=True
    )

    if internet_service == "No":
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
        st.caption("Internet add-ons: No internet service (set automatically)")
    else:
        online_security = st.radio("Online security", ["No", "Yes"], horizontal=True)
        online_backup = st.radio("Online backup", ["No", "Yes"], horizontal=True)
        device_protection = st.radio("Device protection", ["No", "Yes"], horizontal=True)
        tech_support = st.radio("Tech support", ["No", "Yes"], horizontal=True)
        streaming_tv = st.radio("Streaming TV", ["No", "Yes"], horizontal=True)
        streaming_movies = st.radio("Streaming movies", ["No", "Yes"], horizontal=True)

with col3:
    st.write("**Account and billing**")
    contract = st.radio("Contract", category_options["Contract"])
    paperless = st.radio("Paperless billing", category_options["PaperlessBilling"], horizontal=True)
    payment_method = st.radio("Payment method", category_options["PaymentMethod"])
    monthly_charges = st.slider(
        "Monthly charges",
        float(numeric_ranges["MonthlyCharges"]["min"]),
        float(numeric_ranges["MonthlyCharges"]["max"]),
        float(numeric_ranges["MonthlyCharges"]["median"])
    )
    total_charges = st.slider(
        "Total charges",
        float(numeric_ranges["TotalCharges"]["min"]),
        float(numeric_ranges["TotalCharges"]["max"]),
        float(tenure * monthly_charges)
    )

threshold = st.slider("Decision threshold", 0.05, 0.95, default_threshold, step=0.05)
st.caption(f"The default of {default_threshold} came from the threshold experiment in the "
           "notebook, chosen to favour recall. Raise it for an expensive campaign like personal "
           "phone calls, lower it for a cheap one like an automated email.")

if st.button("Analyse this customer"):
    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    encoded = encode_customer(customer)
    churn_prob = model.predict_proba(encoded)[0, 1]

    tab1, tab2, tab3 = st.tabs(["Churn risk", "Revenue at risk", "What to offer"])

    with tab1:
        if churn_prob >= threshold:
            st.write("### This customer is likely to CHURN")
        else:
            st.write("### This customer is likely to STAY")

        st.write("Probability of churn:", f"{churn_prob:.1%}")
        st.progress(float(churn_prob))
        st.caption(f"Using a decision threshold of {threshold}.")

    with tab2:
        # The regression is a straight line fit, so nothing stops it predicting a
        # negative lifespan or one longer than the 72 months in the dataset. Clip it
        # back to the range that actually exists.
        raw_lifespan = lifespan_model.predict(encoded[regression_columns])[0]
        predicted_lifespan = float(np.clip(raw_lifespan, 1, 72))
        months_left = max(predicted_lifespan - tenure, 0)
        revenue_at_risk = churn_prob * monthly_charges * months_left

        st.write("### Revenue at risk: $", round(revenue_at_risk, 2))

        summary = pd.DataFrame([
            {"Measure": "Predicted total lifespan", "Value": f"{predicted_lifespan:.1f} months"},
            {"Measure": "Months already with company", "Value": f"{tenure} months"},
            {"Measure": "Expected months remaining", "Value": f"{months_left:.1f} months"},
            {"Measure": "Monthly charges", "Value": f"${monthly_charges:.2f}"},
            {"Measure": "Churn probability", "Value": f"{churn_prob:.1%}"}
        ])
        st.table(summary)

        st.caption(
            "Revenue at risk = churn probability x monthly charges x expected months remaining. "
            "The lifespan model has an average error of about "
            f"{regression_metrics['MAE']:.1f} months, so treat this as a rough size rather than "
            "an exact figure."
        )

        # Part 20: being at risk is not the same as being worth contacting.
        st.write("---")
        st.write("### Is this customer worth contacting?")

        offer_cost = st.number_input(
            "Cost of the retention offer for this customer ($)",
            min_value=5.0, max_value=500.0, value=float(flat_offer_cost), step=5.0
        )

        expected_gain = churn_prob * monthly_charges * months_left * save_rate

        st.write("Expected gain from contacting them: $", round(expected_gain, 2))
        st.write("Cost of the offer: $", round(offer_cost, 2))

        if expected_gain > offer_cost:
            st.write(f"**Worth contacting.** We expect to gain ${expected_gain - offer_cost:.2f} "
                     "more than the offer costs.")
        else:
            st.write(f"**Not worth contacting.** The offer costs ${offer_cost - expected_gain:.2f} "
                     "more than we expect to get back, even though this customer may still leave.")

        st.caption(
            f"Assumes {save_rate:.0%} of contacted churners actually stay. This is the rule from "
            "Part 20 of the notebook: it beat every single threshold because it uses how much the "
            "customer is worth, not just how likely they are to leave."
        )

    with tab3:
        options = {
            "Move to a 1 year contract": set_category(encoded, "Contract", "Contract_One year"),
            "Move to a 2 year contract": set_category(encoded, "Contract", "Contract_Two year"),
            "Add Tech Support": add_service(encoded, "TechSupport_Yes"),
            "Add Online Security": add_service(encoded, "OnlineSecurity_Yes"),
            "Switch to automatic bank payment": set_category(encoded, "PaymentMethod"),
            "Turn off paperless billing": set_category(encoded, "PaperlessBilling")
        }

        rows = []
        for action, changed in options.items():
            new_risk = model.predict_proba(changed)[0, 1]
            rows.append({
                "Action": action,
                "New risk": new_risk,
                "Risk drop": churn_prob - new_risk
            })

        actions = pd.DataFrame(rows).sort_values("Risk drop", ascending=False).reset_index(drop=True)

        best_action = actions.loc[0, "Action"]
        best_new_risk = actions.loc[0, "New risk"]

        if actions.loc[0, "Risk drop"] <= 0.001:
            st.write("### None of the available offers would help this customer")
            st.caption("They may already be on the best contract and services we can offer.")
        else:
            st.write("### Recommended offer:", best_action)
            st.write(
                "Churn risk would go from", f"{churn_prob:.1%}",
                "down to", f"{best_new_risk:.1%}"
            )

        st.bar_chart(actions.set_index("Action")["Risk drop"])
        st.dataframe(actions.round(3), width="stretch")

        st.caption(
            "The model only learned a correlation, not a cause. Customers on longer contracts "
            "churn less, but that does not guarantee moving this customer onto one will have the "
            "full effect shown here."
        )

st.write("---")
st.write("Dataset: IBM Telco Customer Churn (7043 customers).")
