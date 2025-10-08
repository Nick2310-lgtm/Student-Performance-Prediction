from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data(csv_path, test_size=0.2, random_state=42, cutpoint=12):
    df = pd.read_csv(csv_path, sep=';')
    df['target'] = (df['G3'] >= cutpoint).astype(int)

    X = df.drop(columns=['G1', 'G2', 'G3', 'target'])
    y = df['target']

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values, preprocessor
