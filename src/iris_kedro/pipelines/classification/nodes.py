import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def validate_data(data: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    required = features + [target]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("The raw dataset is empty.")
    return data[required].copy()


def clean_data(data: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    cleaned = data.copy()
    for column in features:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=features + [target])
    cleaned[target] = cleaned[target].astype(str).str.strip()
    cleaned = cleaned[cleaned[target] != ""]
    cleaned = cleaned[(cleaned[features] > 0).all(axis=1)]
    if cleaned.empty or cleaned[target].nunique() < 2:
        raise ValueError("Not enough valid data for classification.")
    return cleaned.reset_index(drop=True)


def split_data(data: pd.DataFrame, features: list[str], target: str,
               test_size: float, random_seed: int) -> tuple:
    return tuple(train_test_split(
        data[features], data[target], test_size=test_size,
        random_state=random_seed, stratify=data[target]
    ))


def train_model(x_train: pd.DataFrame, y_train: pd.Series,
                max_depth: int, random_seed: int) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, x_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    labels = model.classes_.tolist()
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "test_samples": int(len(y_test)),
        "train_samples": int(model.tree_.n_node_samples[0]),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()
    }
