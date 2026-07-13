import pandas as pd
import numpy as np
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Handle missing values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
    # Categorical encoding
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
    if 'Embarked' in df.columns:
        # One-hot encode Embarked
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True).astype(int)
        df = pd.concat([df, embarked_dummies], axis=1)
        df = df.drop(columns=['Embarked'])
        
    return df

def main():
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'titanic_raw', 'titanic.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'titanic_preprocessing')
    output_path = os.path.join(output_dir, 'titanic_cleaned.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(raw_data_path)
    df_clean = preprocess_data(df)
    
    print(f"Saving preprocessed data to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print("Preprocessing finished successfully!")

if __name__ == "__main__":
    main()
