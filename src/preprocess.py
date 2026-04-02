import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Encode categorical "Extracurricular Activities"
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])
    
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler to root or models folder to be used by app.py
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return X_scaled, y