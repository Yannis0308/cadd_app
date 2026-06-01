import pickle 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.linear_model import LogisticRegression, Ridge 
from sklearn.svm import SVC, SVR 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import xgboost as xgb 
import streamlit as st 
import os 
 
def get_available_models(task_type): 
    if task_type == "classification": 
        return { 
            "Random Forest": RandomForestClassifier, 
            "XGBoost": xgb.XGBClassifier, 
            "Logistic Regression": LogisticRegression, 
            "SVM": SVC 
        } 
    else: 
        return { 
            "Random Forest": RandomForestRegressor, 
            "XGBoost": xgb.XGBRegressor, 
            "Ridge Regression": Ridge, 
            "SVR": SVR 
        } 
 
def train_model(model_name, model_class, X_train, y_train, params=None): 
    if params is None: 
        params = {} 
    model = model_class(**params) 
    model.fit(X_train, y_train) 
    return model 
 
def evaluate_classification(model, X_test, y_test): 
    y_pred = model.predict(X_test) 
    metrics = { 
        "Accuracy": accuracy_score(y_test, y_pred), 
        "Precision": precision_score(y_test, y_pred, average='weighted'), 
        "Recall": recall_score(y_test, y_pred, average='weighted'), 
        "F1 Score": f1_score(y_test, y_pred, average='weighted'), 
    } 
    return metrics, y_pred 
 
def evaluate_regression(model, X_test, y_test): 
    y_pred = model.predict(X_test) 
    metrics = { 
        "MSE": mean_squared_error(y_test, y_pred), 
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)), 
        "MAE": mean_absolute_error(y_test, y_pred), 
        "R2": r2_score(y_test, y_pred) 
    } 
    return metrics, y_pred 
 
def save_model(model, model_name, task_type, feature_names): 
    os.makedirs("models/saved_models", exist_ok=True) 
    filename = f"saved_model_{pd.Timestamp.now().strftime('%Y%%m%%d_%%H%%M%%S')}.pkl" 
    filepath = f"models/saved_models/{filename}" 
    with open(filepath, 'wb') as f: 
        pickle.dump({'model': model, 'model_name': model_name, 'task_type': task_type, 'feature_names': feature_names}, f) 
    return filepath 
 
def load_model(filepath): 
    with open(filepath, 'rb') as f: 
        return pickle.load(f) 
 
def predict_new_molecules(model, smiles_list, feature_names): 
    from utils.data_utils import calculate_descriptors 
    features_df, _ = calculate_descriptors(smiles_list) 
    available_features = [f for f in feature_names if f in features_df.columns] 
    features_df = features_df[available_features] 
    predictions = model.predict(features_df) 
    return predictions, features_df 
