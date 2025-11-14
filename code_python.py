#premier depot code python
# ok
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#!/usr/bin/env python3
"""
Code Python minimal pour faire du machine learning.
- Charge un jeu de données (par défaut: Iris)
- Pré-traite (scaling)
- Entraîne plusieurs modèles (LogisticRegression, RandomForest, KNeighbors)
- Évalue (accuracy, classification report, validation croisée)
- Sauvegarde le meilleur modèle

Usage:
    python code_python.py             # utilise Iris et entraîne tous les modèles
    python code_python.py --model rf  # n'entraîne que RandomForest
    python code_python.py --csv data.csv --model lr --save model.joblib
"""


def load_dataset(csv_path: str = None):
    """Charge les données: si csv_path fourni, lit un CSV (dernière colonne = cible),
    sinon charge Iris depuis sklearn."""
    if csv_path:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError("Le CSV doit contenir au moins une colonne de features et une colonne cible.")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    iris = datasets.load_iris()
    return iris.data, iris.target

def preprocess(X_train, X_test):
    """Standardise les features."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def get_models(random_state=42):
    """Retourne un dict de modèles candidates."""
    return {
        "lr": LogisticRegression(max_iter=1000, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5)
    }

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Entraîne, prédit et affiche les métriques. Retourne dict de résultats."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    # cross-val sur l'ensemble d'entraînement
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    result = {
        "name": name,
        "model": model,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores))
    }
    return result

def main(args):
    X, y = load_dataset(args.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    models = get_models(random_state=args.random_state)
    to_run = (args.model.split(',') if args.model != 'all' else list(models.keys()))
    results = []

    for name in to_run:
        if name not in models:
            print(f"Avertissement: modèle inconnu '{name}', ignoré.")
            continue
        res = evaluate_model(name, models[name], X_train_s, y_train, X_test_s, y_test)
        results.append(res)
        print(f"\nModel: {res['name']}")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  CV mean: {res['cv_mean']:.4f}  (std {res['cv_std']:.4f})")
        print("  Classification report:")
        print(res['report'])
        print("  Confusion matrix:")
        print(res['confusion_matrix'])

    if not results:
        print("Aucun modèle entraîné. Fin.")
        return

    # Choisir le meilleur modèle par accuracy sur test set
    best = max(results, key=lambda r: r['accuracy'])
    print(f"\nMeilleur modèle: {best['name']} avec accuracy {best['accuracy']:.4f}")

    if args.save:
        out_path = Path(args.save)
        # Sauvegarder un dict contenant le modèle et le scaler
        artifact = {
            "model_name": best['name'],
            "model": best['model'],
            "scaler": scaler
        }
        joblib.dump(artifact, out_path)
        print(f"Modèle sauvegardé dans: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script ML simple")
    parser.add_argument("--csv", type=str, default=None, help="Chemin vers un CSV (dernière colonne = cible).")
    parser.add_argument("--model", type=str, default="all", help="Quel(s) modèle(s) entraîner: lr,rf,knn ou 'all' (comma séparés).")
    parser.add_argument("--save", type=str, default=None, help="Chemin pour sauvegarder le meilleur modèle (.joblib).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Taille du test set (fraction).")
    parser.add_argument("--random-state", type=int, default=42, help="Seed pour reproductibilité.")
    args = parser.parse_args()
    main(args)