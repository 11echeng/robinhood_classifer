# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
from optuna.integration import SklearnPruningCallback
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.base import TransformerMixin, BaseEstimator
import warnings

warnings.filterwarnings('ignore')

# Define DenseTransformer to convert sparse matrix to dense
class DenseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()

# Define PyTorch Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(NeuralNet, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Load Data
def load_data():
    # Replace this with your actual data loading code
    # df = pd.read_csv('your_dataset.csv')
    # For demonstration, let's assume df is provided
    # Ensure df is defined before this function is called
    return df.copy()

# Preprocess Data
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop('user_id', axis=1)

    # Define target and features
    y = df['churn_flag']
    X = df.drop('churn_flag', axis=1)

    # Identify categorical and numerical columns
    categorical_cols = [
        'risk_tolerance',
        'investment_experience',
        'liquidity_needs',
        'platform',
        'instrument_type_first_traded',
        'time_horizon'
    ]
    numerical_cols = ['time_spent', 'first_deposit_amount']

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor

# Handle Class Imbalance
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Define Model Training and Evaluation Pipeline
def train_evaluate_model(X_train, y_train, X_test, y_test, clf, model_name):
    # Fit the model
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"{model_name} ROC AUC Score: {roc_auc:.4f}")
    print(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")

    return clf, roc_auc

# Hyperparameter Tuning with Optuna
def hyperparameter_tuning(X_train, y_train, preprocessor, model_name):
    def objective(trial):
        if model_name == 'LogisticRegression':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42)
        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        elif model_name == 'XGBoost':
            param = {
                'verbosity': 0,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = xgb.XGBClassifier(**param)
        elif model_name == 'LightGBM':
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            }
            model = lgb.LGBMClassifier(**param)
        elif model_name == 'CatBoost':
            param = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }
            model = CatBoostClassifier(**param, silent=True, random_state=42)
        elif model_name == 'PyTorch':
            # Hyperparameters to tune
            num_layers = trial.suggest_int('num_layers', 1, 3)
            hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
            max_epochs = trial.suggest_int('max_epochs', 10, 50, step=10)

            # Fit the preprocessor to get input size
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            input_size = X_train_processed.shape[1]

            # Define the PyTorch model
            module = NeuralNet(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )

            # Use skorch NeuralNetClassifier
            model = NeuralNetClassifier(
                module=module,
                max_epochs=max_epochs,
                lr=lr,
                batch_size=batch_size,
                optimizer=optim.Adam,
                criterion=nn.BCELoss,
                iterator_train__shuffle=True,
                verbose=0,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Create pipeline
            clf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('to_dense', DenseTransformer()),
                ('classifier', model)
            ])
        else:
            return 0  # Invalid model name

        if model_name != 'PyTorch':
            # Create pipeline
            clf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            clf.fit(X_tr, y_tr)
            y_proba = clf.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print(f"Best Trial: {study.best_trial.params}")

    # Return the best model
    best_params = study.best_trial.params
    if model_name == 'LogisticRegression':
        penalty = best_params['penalty']
        C = best_params['C']
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42)
    elif model_name == 'RandomForest':
        best_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
    elif model_name == 'XGBoost':
        best_params['verbosity'] = 0
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
        best_params['use_label_encoder'] = False
        best_params['booster'] = 'gbtree'
        best_model = xgb.XGBClassifier(**best_params)
    elif model_name == 'LightGBM':
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        best_model = lgb.LGBMClassifier(**best_params)
    elif model_name == 'CatBoost':
        best_model = CatBoostClassifier(**best_params, silent=True, random_state=42)
    elif model_name == 'PyTorch':
        num_layers = best_params['num_layers']
        hidden_size = best_params['hidden_size']
        dropout_rate = best_params['dropout_rate']
        lr = best_params['lr']
        batch_size = best_params['batch_size']
        max_epochs = best_params['max_epochs']

        # Fit the preprocessor to get input size
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        input_size = X_train_processed.shape[1]

        module = NeuralNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        best_model = NeuralNetClassifier(
            module=module,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer=optim.Adam,
            criterion=nn.BCELoss,
            iterator_train__shuffle=True,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Create pipeline with DenseTransformer
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('to_dense', DenseTransformer()),
            ('classifier', best_model)
        ])
        return clf
    else:
        best_model = None

    # Create pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    return clf

# Main Function
def main():
    # Load and preprocess data
    df = load_data()
    X, y, preprocessor = preprocess_data(df)

    # Handle class imbalance
    X_resampled, y_resampled = balance_data(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Models to train
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(silent=True, random_state=42),
        'PyTorch': 'PyTorch'  # Placeholder for model name
    }

    # Dictionary to store results
    model_results = {}

    for model_name in models.keys():
        print(f"\nTraining {model_name}...")

        # Hyperparameter tuning
        print("Performing hyperparameter tuning...")
        best_model_pipeline = hyperparameter_tuning(X_train, y_train, preprocessor, model_name)

        # Train and evaluate model
        print("Training and evaluating model...")
        clf, roc_auc = train_evaluate_model(
            X_train, y_train, X_test, y_test, best_model_pipeline, model_name
        )

        # Store the trained pipeline and performance
        model_results[model_name] = {
            'pipeline': clf,
            'roc_auc': roc_auc
        }

    # Select the best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['roc_auc'])
    print(f"\nBest Model: {best_model_name} with ROC AUC: {model_results[best_model_name]['roc_auc']:.4f}")

    # Predict churn probabilities on the original dataset
    print("\nPredicting churn probabilities on the original dataset...")
    best_pipeline = model_results[best_model_name]['pipeline']
    churn_probabilities = best_pipeline.predict_proba(X)[:, 1]
    df['churn_probability'] = churn_probabilities
    df['predicted_churn'] = (df['churn_probability'] >= 0.5).astype(int)

    # Display predictions
    print(df[['churn_flag', 'churn_probability', 'predicted_churn']].head())

    # Save the best model
    print("\nSaving the best model...")
    import joblib
    joblib.dump(best_pipeline, f'best_model_{best_model_name}.pkl')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
