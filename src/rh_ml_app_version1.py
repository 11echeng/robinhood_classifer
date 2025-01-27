'''
author: chatGPT
'''
# Import necessary libraries
import pandas as pd
import numpy as np

# Data Manipulation and Handling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Machine Learning Libraries
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter Optimization
import optuna

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42


def preprocess_data(df):
    """
    Preprocess the data by separating features and target,
    and creating a preprocessor pipeline.
    """
    # Separate features and target variable
    X = df.drop(['user_id', 'churn_flag'], axis=1)
    y = df['churn_flag']

    # Identify categorical and numerical columns
    categorical_cols = [
        'risk_tolerance',
        'investment_experience',
        'liquidity_needs',
        'platform',
        'instrument_type_first_traded',
        'time_horizon',
    ]
    numerical_cols = ['time_spent', 'first_deposit_amount']

    # Define preprocessing steps
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    return X, y, preprocessor


def get_model_pipeline(preprocessor, model):
    """
    Create a pipeline that includes preprocessing,
    SMOTE oversampling, and the model.
    """
    pipeline = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model),
        ]
    )
    return pipeline


def objective_rf(trial, X, y, preprocessor):
    """
    Objective function for Random Forest hyperparameter tuning using Optuna.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical(
            'max_features', ['auto', 'sqrt', 'log2']
        ),
    }
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    pipeline = get_model_pipeline(preprocessor, model)

    # Use cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return np.mean(scores)


def objective_logistic(trial, X, y, preprocessor):
    """
    Objective function for Logistic Regression hyperparameter tuning using Optuna.
    """
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    pipeline = get_model_pipeline(preprocessor, model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return np.mean(scores)


def objective_xgb(trial, X, y, preprocessor):
    """
    Objective function for XGBoost hyperparameter tuning using Optuna.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
    )
    pipeline = get_model_pipeline(preprocessor, model)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return np.mean(scores)


def objective_lgb(trial, X, y, preprocessor):
    """
    Objective function for LightGBM hyperparameter tuning using Optuna.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE)
    pipeline = get_model_pipeline(preprocessor, model)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return np.mean(scores)


def run_study(objective, X, y, preprocessor, n_trials=20):
    """
    Run an Optuna study for hyperparameter tuning.
    """
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, X, y, preprocessor)
    study.optimize(func, n_trials=n_trials)
    return study


def evaluate_model(model, X_train, y_train, X_test, y_test, preprocessor):
    """
    Evaluate the model on the test set and print metrics.
    """
    # Create pipeline
    pipeline = get_model_pipeline(preprocessor, model)

    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)

    # Predict probabilities and labels
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # Compute ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')  # Random classifier line
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    return pipeline, roc_auc


def main():
    # Load data
    df = pd.read_csv('your_dataset.csv')  # Replace with your dataset path

    # Preprocess data
    X, y, preprocessor = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Run studies for each model
    print("Running hyperparameter tuning for Random Forest...")
    study_rf = run_study(objective_rf, X_train, y_train, preprocessor)
    print(f"Best ROC AUC for Random Forest: {study_rf.best_value:.4f}")
    print(f"Best parameters: {study_rf.best_params}\n")

    print("Running hyperparameter tuning for Logistic Regression...")
    study_logistic = run_study(
        objective_logistic, X_train, y_train, preprocessor
    )
    print(
        f"Best ROC AUC for Logistic Regression: {study_logistic.best_value:.4f}"
    )
    print(f"Best parameters: {study_logistic.best_params}\n")

    print("Running hyperparameter tuning for XGBoost...")
    study_xgb = run_study(objective_xgb, X_train, y_train, preprocessor)
    print(f"Best ROC AUC for XGBoost: {study_xgb.best_value:.4f}")
    print(f"Best parameters: {study_xgb.best_params}\n")

    print("Running hyperparameter tuning for LightGBM...")
    study_lgb = run_study(objective_lgb, X_train, y_train, preprocessor)
    print(f"Best ROC AUC for LightGBM: {study_lgb.best_value:.4f}")
    print(f"Best parameters: {study_lgb.best_params}\n")

    # Compare models
    best_studies = {
        'Random Forest': study_rf,
        'Logistic Regression': study_logistic,
        'XGBoost': study_xgb,
        'LightGBM': study_lgb,
    }
    # Select best model
    best_model_name = max(best_studies, key=lambda k: best_studies[k].best_value)
    print(f"Best model: {best_model_name}\n")

    # Retrieve the best parameters and model
    if best_model_name == 'Random Forest':
        best_params = study_rf.best_params
        model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    elif best_model_name == 'Logistic Regression':
        best_params = study_logistic.best_params
        penalty = best_params['penalty']
        C = best_params['C']
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
    elif best_model_name == 'XGBoost':
        best_params = study_xgb.best_params
        model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
        )
    elif best_model_name == 'LightGBM':
        best_params = study_lgb.best_params
        model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE)
    else:
        raise ValueError("Invalid model name")

    # Evaluate the best model
    pipeline, roc_auc = evaluate_model(
        model, X_train, y_train, X_test, y_test, preprocessor
    )
    print(
        f"Test ROC AUC of the best model ({best_model_name}): {roc_auc:.4f}\n"
    )

    # Predict churn probabilities on the entire dataset
    pipeline.fit(X_train, y_train)
    y_pred_proba_all = pipeline.predict_proba(X)[:, 1]
    y_pred_all = pipeline.predict(X)

    # Add predictions to the original DataFrame
    df['churn_probability'] = y_pred_proba_all
    df['predicted_churn'] = y_pred_all

    # Save the model if needed
    # import joblib
    # joblib.dump(pipeline, 'best_model_pipeline.pkl')

    # Optional: Model interpretability using SHAP (for tree-based models)
    if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        # Preprocess data
        X_processed = preprocessor.fit_transform(X)
        # Fit model on the entire dataset
        model.fit(X_processed, y)
        # Compute SHAP values
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        # Plot SHAP summary plot
        shap.summary_plot(shap_values, X_processed)

    # The final DataFrame df now contains 'churn_probability' and 'predicted_churn'
    # You can save df to a CSV file if needed
    # df.to_csv('churn_predictions.csv', index=False)


if __name__ == '__main__':
    main()
