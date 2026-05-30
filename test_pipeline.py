# Unit Testing Suite for Churn Prediction ML Pipeline
import unittest
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from train_model import generate_synthetic_data, serialize_tree

class TestMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Generate synthetic data for general testing to conserve CPU
        cls.df = generate_synthetic_data(n_samples=200, random_seed=42)
        
    def test_data_synthesis(self):
        """Test dataset size, shapes, column existence, and ranges."""
        # Test 1000 records generation
        full_df = generate_synthetic_data(n_samples=1000, random_seed=42)
        self.assertEqual(full_df.shape[0], 1000)
        self.assertEqual(full_df.shape[1], 13)
        
        # Test column existence
        expected_cols = [
            "CustomerID", "SeniorCitizen", "Partner", "Tenure", "Contract",
            "InternetService", "TechSupport", "MonthlyCharges", "TotalCharges",
            "SupportTickets", "PaymentMethod", "PaperlessBilling", "Churn"
        ]
        for col in expected_cols:
            self.assertIn(col, full_df.columns)
            
        # Test absence of null values
        self.assertEqual(full_df.isnull().sum().sum(), 0)
        
        # Test specific ranges
        self.assertTrue((full_df["Tenure"] >= 1).all() and (full_df["Tenure"] <= 72).all())
        self.assertTrue((full_df["SupportTickets"] >= 0).all() and (full_df["SupportTickets"] <= 9).all())
        self.assertTrue((full_df["SeniorCitizen"].isin([0, 1])).all())
        self.assertTrue((full_df["Churn"].isin([0, 1])).all())
        
    def test_preprocessing(self):
        """Verify that categoricals correctly map to numeric binary encodings."""
        df_encoded = pd.DataFrame()
        df_encoded["Tenure"] = self.df["Tenure"]
        df_encoded["MonthlyCharges"] = self.df["MonthlyCharges"]
        df_encoded["SupportTickets"] = self.df["SupportTickets"]
        df_encoded["SeniorCitizen"] = self.df["SeniorCitizen"]
        
        # Manually verify manual one-hot encoding logic
        df_encoded["Contract_MonthToMonth"] = (self.df["Contract"] == "Month-to-month").astype(int)
        df_encoded["Contract_TwoYear"] = (self.df["Contract"] == "Two year").astype(int)
        df_encoded["Internet_FiberOptic"] = (self.df["InternetService"] == "Fiber optic").astype(int)
        df_encoded["TechSupport_No"] = (self.df["TechSupport"] == "No").astype(int)
        df_encoded["Payment_ElectronicCheck"] = (self.df["PaymentMethod"] == "Electronic check").astype(int)
        df_encoded["PaperlessBilling_Yes"] = (self.df["PaperlessBilling"] == "Yes").astype(int)
        df_encoded["Partner_Yes"] = (self.df["Partner"] == "Yes").astype(int)
        
        # Test dimensions
        self.assertEqual(df_encoded.shape[0], self.df.shape[0])
        self.assertEqual(df_encoded.shape[1], 11)
        
        # Test values are strictly binary
        binary_cols = [
            "Contract_MonthToMonth", "Contract_TwoYear", "Internet_FiberOptic",
            "TechSupport_No", "Payment_ElectronicCheck", "PaperlessBilling_Yes", "Partner_Yes"
        ]
        for col in binary_cols:
            self.assertTrue(df_encoded[col].isin([0, 1]).all())
            
    def test_model_fit(self):
        """Assert both Logistic Regression and Decision Tree models fit and surpass minimum baseline accuracy (>70%)."""
        full_df = generate_synthetic_data(n_samples=1000, random_seed=42)
        X = pd.DataFrame()
        X["Tenure"] = full_df["Tenure"]
        X["MonthlyCharges"] = full_df["MonthlyCharges"]
        X["SupportTickets"] = full_df["SupportTickets"]
        X["SeniorCitizen"] = full_df["SeniorCitizen"]
        X["Partner_Yes"] = (full_df["Partner"] == "Yes").astype(int)
        X["Contract_MonthToMonth"] = (full_df["Contract"] == "Month-to-month").astype(int)
        X["Contract_TwoYear"] = (full_df["Contract"] == "Two year").astype(int)
        X["Internet_FiberOptic"] = (full_df["InternetService"] == "Fiber optic").astype(int)
        X["TechSupport_No"] = (full_df["TechSupport"] == "No").astype(int)
        X["Payment_ElectronicCheck"] = (full_df["PaymentMethod"] == "Electronic check").astype(int)
        X["PaperlessBilling_Yes"] = (full_df["PaperlessBilling"] == "Yes").astype(int)
        
        y = full_df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and test Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_acc = lr.score(X_test, y_test)
        self.assertGreater(lr_acc, 0.70)
        
        # Train and test Decision Tree
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(X_train, y_train)
        dt_acc = dt.score(X_test, y_test)
        self.assertGreater(dt_acc, 0.70)
        
    def test_tree_serialization(self):
        """Confirm that the Decision Tree serializer outputs all structural nodes and leaves."""
        X = pd.DataFrame()
        X["Tenure"] = self.df["Tenure"]
        X["MonthlyCharges"] = self.df["MonthlyCharges"]
        X["SupportTickets"] = self.df["SupportTickets"]
        X["SeniorCitizen"] = self.df["SeniorCitizen"]
        X["Partner_Yes"] = (self.df["Partner"] == "Yes").astype(int)
        X["Contract_MonthToMonth"] = (self.df["Contract"] == "Month-to-month").astype(int)
        X["Contract_TwoYear"] = (self.df["Contract"] == "Two year").astype(int)
        X["Internet_FiberOptic"] = (self.df["InternetService"] == "Fiber optic").astype(int)
        X["TechSupport_No"] = (self.df["TechSupport"] == "No").astype(int)
        X["Payment_ElectronicCheck"] = (self.df["PaymentMethod"] == "Electronic check").astype(int)
        X["PaperlessBilling_Yes"] = (self.df["PaperlessBilling"] == "Yes").astype(int)
        
        y = self.df["Churn"]
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X, y)
        
        feature_names = X.columns.tolist()
        serialized = serialize_tree(dt.tree_, feature_names)
        
        # Check standard properties
        self.assertIsInstance(serialized, dict)
        self.assertIn("type", serialized)
        self.assertTrue(serialized["type"] in ["node", "leaf"])
        
        if serialized["type"] == "node":
            self.assertIn("feature", serialized)
            self.assertIn("threshold", serialized)
            self.assertIn("left", serialized)
            self.assertIn("right", serialized)
            self.assertIsInstance(serialized["left"], dict)
            self.assertIsInstance(serialized["right"], dict)
        else:
            self.assertIn("probability", serialized)
            self.assertIn("samples", serialized)
            
if __name__ == "__main__":
    unittest.main()
