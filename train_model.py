import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def generate_synthetic_data(n_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Generate basic identifiers
    customer_ids = [f"US-{1000 + i}" for i in range(n_samples)]
    
    # 2. Demographics: SeniorCitizen (20% prob) & Partner (45% prob)
    senior_citizen = np.random.choice([1, 0], size=n_samples, p=[0.20, 0.80])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])
    
    # 3. Tenure (months on contract, 1 to 72)
    tenure = np.random.randint(1, 73, size=n_samples)
    
    # 4. Internet Service & Tech Support
    internet_options = ["DSL", "Fiber optic", "No"]
    internet_service = np.random.choice(internet_options, size=n_samples, p=[0.35, 0.45, 0.20])
    
    tech_support = []
    for internet in internet_service:
        if internet == "No":
            tech_support.append("No")
        else:
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
    tech_support = np.array(tech_support)
    
    # 5. Contract Type
    contract_options = ["Month-to-month", "One year", "Two year"]
    contract_type = np.random.choice(contract_options, size=n_samples, p=[0.55, 0.20, 0.25])
    
    # 6. Monthly Charges (based on internet service)
    monthly_charges = []
    for internet in internet_service:
        if internet == "Fiber optic":
            charge = np.random.uniform(70.0, 115.0)
        elif internet == "DSL":
            charge = np.random.uniform(45.0, 75.0)
        else: # No internet
            charge = np.random.uniform(18.0, 30.0)
        monthly_charges.append(round(charge, 2))
    monthly_charges = np.array(monthly_charges)
    
    # 7. Total Charges
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 15, size=n_samples), 2)
    total_charges = np.clip(total_charges, 18.0, None)
    
    # 8. Support Tickets (Poisson distribution, higher for monthly contracts and no support)
    support_tickets = []
    for ct, ts in zip(contract_type, tech_support):
        if ct == "Month-to-month" and ts == "No":
            lam = 2.2
        elif ct == "Month-to-month" or ts == "No":
            lam = 1.2
        else:
            lam = 0.4
        tickets = np.random.poisson(lam)
        support_tickets.append(min(tickets, 9)) # Cap at 9 tickets
    support_tickets = np.array(support_tickets)
    
    # 9. Payment Method & Paperless Billing
    payment_options = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_method = np.random.choice(payment_options, size=n_samples, p=[0.40, 0.20, 0.20, 0.20])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    
    # 10. Logistic Sigmoid Model for Churn Probability
    # Calibrated weights to generate clean, highly-explainable churn indicators
    z = (
        0.35 + 
        (contract_type == "Month-to-month").astype(int) * 1.30 +
        (contract_type == "Two year").astype(int) * -0.85 +
        (internet_service == "Fiber optic").astype(int) * 0.40 +
        (tech_support == "No").astype(int) * 0.65 +
        support_tickets * 0.50 +
        (payment_method == "Electronic check").astype(int) * 0.45 +
        (paperless_billing == "Yes").astype(int) * 0.20 +
        senior_citizen * 0.35 -
        (partner == "Yes").astype(int) * 0.30 -
        tenure * 0.052 +
        (monthly_charges - 60.0) * 0.007 -
        1.55 # Basal offset
    )
    
    prob = 1 / (1 + np.exp(-z))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    # 11. Assemble into DataFrame
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Tenure": tenure,
        "Contract": contract_type,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "Churn": churn
    })
    
    return df

def serialize_tree(tree, feature_names, node_id=0):
    """Recursively serialize a Scikit-Learn Decision Tree into a nested JSON dictionary."""
    if tree.feature[node_id] == -2: # Leaf node indicator
        values = tree.value[node_id][0]
        prob = float(values[1] / sum(values))
        return {
            "type": "leaf",
            "probability": round(prob, 4),
            "samples": int(sum(values))
        }
    else:
        feature_index = tree.feature[node_id]
        feature_name = feature_names[feature_index]
        threshold = float(tree.threshold[node_id])
        return {
            "type": "node",
            "feature": feature_name,
            "threshold": round(threshold, 4),
            "left": serialize_tree(tree, feature_names, tree.children_left[node_id]),
            "right": serialize_tree(tree, feature_names, tree.children_right[node_id])
        }

def main():
    print("=== Phase 1: Upgraded Dual-Model ML Pipeline ===")
    print("Synthesizing expanded customer churn dataset...")
    df = generate_synthetic_data(n_samples=1000)
    
    # Save dataset
    csv_path = "customer_churn_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path} (shape: {df.shape})")
    print(f"Overall Churn Rate: {df['Churn'].mean() * 100:.2f}%")
    
    # --- Feature Engineering for Model Training ---
    print("\nPreparing features for Scikit-Learn training...")
    X_raw = df.copy()
    y = X_raw["Churn"]
    
    X = pd.DataFrame()
    X["Tenure"] = X_raw["Tenure"]
    X["MonthlyCharges"] = X_raw["MonthlyCharges"]
    X["SupportTickets"] = X_raw["SupportTickets"]
    
    # Demographics
    X["SeniorCitizen"] = X_raw["SeniorCitizen"]
    X["Partner_Yes"] = (X_raw["Partner"] == "Yes").astype(int)
    
    # Categoricals
    X["Contract_MonthToMonth"] = (X_raw["Contract"] == "Month-to-month").astype(int)
    X["Contract_TwoYear"] = (X_raw["Contract"] == "Two year").astype(int)
    X["Internet_FiberOptic"] = (X_raw["InternetService"] == "Fiber optic").astype(int)
    X["TechSupport_No"] = (X_raw["TechSupport"] == "No").astype(int)
    X["Payment_ElectronicCheck"] = (X_raw["PaymentMethod"] == "Electronic check").astype(int)
    X["PaperlessBilling_Yes"] = (X_raw["PaperlessBilling"] == "Yes").astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------------- MODEL 1: Logistic Regression ----------------
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)
    
    print("--- Logistic Regression Metrics ---")
    print(f"Accuracy:  {lr_accuracy * 100:.2f}% | Precision: {lr_precision * 100:.2f}% | Recall: {lr_recall * 100:.2f}% | ROC-AUC: {lr_auc:.4f}")
    
    # Extract coefficients
    feature_names = X.columns.tolist()
    coefficients = lr_model.coef_[0].tolist()
    intercept = float(lr_model.intercept_[0])
    
    # ---------------- MODEL 2: Decision Tree ----------------
    print("\nTraining Decision Tree Classifier (max_depth=4)...")
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    
    dt_pred = dt_model.predict(X_test)
    dt_prob = dt_model.predict_proba(X_test)[:, 1]
    
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred)
    dt_recall = recall_score(y_test, dt_pred)
    dt_auc = roc_auc_score(y_test, dt_prob)
    
    print("--- Decision Tree Metrics ---")
    print(f"Accuracy:  {dt_accuracy * 100:.2f}% | Precision: {dt_precision * 100:.2f}% | Recall: {dt_recall * 100:.2f}% | ROC-AUC: {dt_auc:.4f}")
    
    # Extract feature importances
    importances = dt_model.feature_importances_.tolist()
    tree_importances = dict(zip(feature_names, importances))
    
    # Serialize the decision tree structure
    serialized_tree = serialize_tree(dt_model.tree_, feature_names)
    
    # --- Generate Stats & Baseline Segment Risks ---
    avg_tenure = float(df["Tenure"].mean())
    avg_charges = float(df["MonthlyCharges"].mean())
    avg_tickets = float(df["SupportTickets"].mean())
    
    churn_by_contract = df.groupby("Contract")["Churn"].mean().to_dict()
    churn_by_tech_support = df.groupby("TechSupport")["Churn"].mean().to_dict()
    churn_by_internet = df.groupby("InternetService")["Churn"].mean().to_dict()
    charges_by_churn = df.groupby("Churn")["MonthlyCharges"].mean().to_dict()
    
    # Assemble comprehensive Javascript model bundle
    js_model_data = {
        "lr_model": {
            "intercept": intercept,
            "coefficients": dict(zip(feature_names, coefficients)),
            "metrics": {
                "accuracy": round(lr_accuracy, 4),
                "precision": round(lr_precision, 4),
                "recall": round(lr_recall, 4),
                "auc": round(lr_auc, 4)
            }
        },
        "dt_model": {
            "decision_tree": serialized_tree,
            "importances": tree_importances,
            "metrics": {
                "accuracy": round(dt_accuracy, 4),
                "precision": round(dt_precision, 4),
                "recall": round(dt_recall, 4),
                "auc": round(dt_auc, 4)
            }
        },
        "averages": {
            "tenure": round(avg_tenure, 2),
            "monthly_charges": round(avg_charges, 2),
            "support_tickets": round(avg_tickets, 2),
            "total_samples": len(df),
            "churn_rate": round(float(df["Churn"].mean()), 4)
        },
        "segments": {
            "churn_by_contract": {k: round(float(v), 4) for k, v in churn_by_contract.items()},
            "churn_by_tech_support": {k: round(float(v), 4) for k, v in churn_by_tech_support.items()},
            "churn_by_internet": {k: round(float(v), 4) for k, v in churn_by_internet.items()},
            "charges_by_churn": {int(k): round(float(v), 2) for k, v in charges_by_churn.items()}
        }
    }
    
    js_content = f"""// Model parameters and dataset statistics generated by train_model.py using Scikit-Learn
const SCIKIT_LEARN_MODEL_DATA = {json.dumps(js_model_data, indent=2)};

// Expose data if running in Node.js environment for testing, otherwise it is global in browser
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = SCIKIT_LEARN_MODEL_DATA;
}}
"""
    
    js_path = "model_data.js"
    with open(js_path, "w") as f:
        f.write(js_content)
    print(f"\nModel coefficients and decision tree structure exported successfully to: {js_path}")
    print("=========================================================")

if __name__ == "__main__":
    main()
