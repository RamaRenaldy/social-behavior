import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# --- PERUBAHAN 1: Import KNN dan Hapus LGBM ---
from sklearn.neighbors import KNeighborsClassifier  # Mengganti LGBM
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Mengabaikan warning agar output lebih bersih
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("File not found. Please ensure 'personality_dataset.csv' is in the directory/dataset folder.")
        return None, None, None, None, None

    # Pisahkan Fitur dan Target
    X = df.drop('Personality', axis=1)
    y = df['Personality']

    # --- Handling Missing Values & Encoding ---
    print(df.isnull().sum())
    # Pisahkan numerik dan kategorikal
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Imputasi
    imputer_num = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

    # Encoding Kategorikal (Fitur)
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encoding Target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # # Scaling Numerik
    # scaler = StandardScaler()
    # X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Simpan semua preprocessor
    preprocessors = {
        'encoders': encoders,
        'target_encoder': le_target,
        # 'scaler': scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'imputer_num': imputer_num,
        'imputer_cat': imputer_cat
    }

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessors

def train_and_tune(X_train, X_test, y_train, y_test):
    print("\n--- Training & Tuning Models ---")
    
    models_params = {
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {'var_smoothing': np.logspace(0, -9, num=10)}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }

    results = {}
    best_estimators = {}
    
    comparison_history = [] 

    for name, config in models_params.items():
        print(f"Processing: {name}...")
        
        # 1. Baseline (Tanpa Tuning)
        base_model = config['model']
        base_model.fit(X_train, y_train)
        y_pred_base = base_model.predict(X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        # 2. Tuning (GridSearch)
        grid = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred_tuned = best_model.predict(X_test)
        acc_tuned = accuracy_score(y_test, y_pred_tuned)
        
        # Simpan hasil terbaik
        results[name] = acc_tuned
        best_estimators[name] = best_model
        
        # Simpan riwayat untuk ditampilkan di Streamlit
        comparison_history.append({
            'Model': name,
            'Akurasi Baseline': acc_base,
            'Akurasi Tuned': acc_tuned,
            'Improvement': acc_tuned - acc_base
        })
        
        print(f"   > Baseline: {acc_base:.2%} | Tuned: {acc_tuned:.2%}")

    return results, best_estimators, comparison_history

if __name__ == "__main__":
    # Sesuaikan path dataset Anda
    csv_file = 'dataset/personality_dataset.csv' # Pastikan path ini benar
    
    # 1. Load
    X_train, X_test, y_train, y_test, preprocessors = load_and_preprocess_data(csv_file)
    
    if X_train is not None:
        # 2. Train
        results, best_estimators, history = train_and_tune(X_train, X_test, y_train, y_test)
        
        # 3. Pilih Terbaik
        best_model_name = max(results, key=results.get)
        best_model_overall = best_estimators[best_model_name]
        best_acc_overall = results[best_model_name]
        
        print(f"\n🏆 Model Terbaik: {best_model_name} ({best_acc_overall:.2%})")
        
        # 4. Simpan Artifact
        print("\n--- Saving Artifacts ---")
        artifact = {
            'model': best_model_overall,
            'model_name': best_model_name,
            'accuracy': best_acc_overall,
            'preprocessors': preprocessors,
            'all_models': best_estimators,
            'history': history  # --- PERUBAHAN 3: Menyimpan history ke file ---
        }
        
        filename = 'new_personality_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(artifact, f)
            
        print(f"✅ Selesai! File '{filename}' berhasil disimpan.")