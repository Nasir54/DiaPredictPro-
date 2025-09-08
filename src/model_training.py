from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

def train_models(X_train, y_train):
    """Train multiple models and return the best one"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVC': SVC(random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        
        if train_score > best_score:
            best_score = train_score
            best_model = model
            
        print(f"{name} - Training Accuracy: {train_score:.4f}")
    
    return best_model

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, scaler, model_path='../models/diabetes_model.pkl', 
               scaler_path='../models/scaler.pkl'):
    """Save trained model and scaler"""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    # Load and preprocess data
    df = load_data('../data/raw/diabetes.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train and evaluate model
    model = train_models(X_train, y_train)
    tuned_model = hyperparameter_tuning(X_train, y_train)
    accuracy = evaluate_model(tuned_model, X_test, y_test)
    
    # Save the best model
    save_model(tuned_model, scaler)